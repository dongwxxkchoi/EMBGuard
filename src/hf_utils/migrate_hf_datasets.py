#!/usr/bin/env python3
"""
Migrate EMBGuard Hugging Face datasets to the full-name type schema.

This script reads existing Hub datasets and pushes transformed DatasetDicts.
It does not require local CSV/image files; the Hub dataset is the source.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.test_types import (  # noqa: E402
    TEST_TYPE_CODES,
    hf_split_candidates,
    normalize_test_type_code,
    test_type_label,
    test_type_slug,
)


DATASET_ALIASES = {
    "embguardtest": "EMBGuardTest",
    "embguardtest_v2": "EMBGuardTest_v2",
    "embguardtest-v2": "EMBGuardTest_v2",
    "test": "EMBGuardTest_v2",
    "test_v2": "EMBGuardTest_v2",
    "test-v2": "EMBGuardTest_v2",
    "heldout": "heldout_set",
    "heldout_set": "heldout_set",
    "embhazard": "EMBHazard",
    "hazard": "EMBHazard",
}


def canonical_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name.strip().lower(), name.strip())


def _load_dataset_dict(repo_id: str, token: Optional[str], revision: Optional[str]) -> DatasetDict:
    print(f"Loading source dataset: {repo_id}")
    dataset = load_dataset(repo_id, token=token, revision=revision)
    if isinstance(dataset, DatasetDict):
        return dataset
    return DatasetDict({"train": dataset})


def _label_for_value(value: object) -> str:
    code = normalize_test_type_code(value)
    return test_type_label(code) if code in TEST_TYPE_CODES else ""


def _add_type_label_columns(dataset: Dataset, force_type_code: Optional[str] = None) -> Dataset:
    has_type = "Type" in dataset.column_names
    has_subtype = "Subtype" in dataset.column_names

    # Avoid dataset.map() here: rewriting large image columns can overflow Arrow
    # offsets for image-heavy datasets. Adding/replacing scalar columns is safer.
    if "Type Label" in dataset.column_names:
        dataset = dataset.remove_columns(["Type Label"])
    if "Subtype Label" in dataset.column_names:
        dataset = dataset.remove_columns(["Subtype Label"])

    if force_type_code:
        if "Type" in dataset.column_names:
            dataset = dataset.remove_columns(["Type"])
        dataset = dataset.add_column("Type", [force_type_code] * len(dataset))
        dataset = dataset.add_column("Type Label", [test_type_label(force_type_code)] * len(dataset))
    elif has_type:
        dataset = dataset.add_column("Type Label", [_label_for_value(value) for value in dataset["Type"]])

    if has_subtype:
        dataset = dataset.add_column("Subtype Label", [_label_for_value(value) for value in dataset["Subtype"]])

    return dataset


def migrate_embguardtest(source: DatasetDict) -> DatasetDict:
    migrated = DatasetDict()
    for code in TEST_TYPE_CODES:
        source_split = None
        for candidate in hf_split_candidates(code):
            if candidate in source:
                source_split = candidate
                break

        if not source_split:
            print(f"Warning: no source split found for {code}; tried {hf_split_candidates(code)}")
            continue

        target_split = test_type_slug(code)
        print(f"  {source_split} -> {target_split} ({test_type_label(code)})")
        migrated[target_split] = _add_type_label_columns(source[source_split], force_type_code=code)

    if not migrated:
        raise ValueError("No EMBGuardTest splits were found to migrate.")

    return migrated


def migrate_label_columns_only(source: DatasetDict) -> DatasetDict:
    migrated = DatasetDict()
    for split_name, dataset in source.items():
        print(f"  {split_name} -> {split_name}")
        migrated[split_name] = _add_type_label_columns(dataset)
    return migrated


def migrate_dataset(dataset_name: str, source: DatasetDict) -> DatasetDict:
    if dataset_name.startswith("EMBGuardTest"):
        return migrate_embguardtest(source)
    if dataset_name in {"heldout_set", "EMBHazard"}:
        return migrate_label_columns_only(source)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def print_dataset_summary(dataset_name: str, dataset_dict: DatasetDict) -> None:
    print(f"\n{dataset_name} migrated summary:")
    for split_name, dataset in dataset_dict.items():
        print(f"  - {split_name}: {len(dataset)} rows")
        print(f"    columns: {dataset.column_names}")
        if len(dataset) > 0:
            sample = dataset[0]
            sample_bits = []
            for key in ["Type", "Type Label", "Subtype", "Subtype Label"]:
                if key in sample:
                    sample_bits.append(f"{key}={sample[key]!r}")
            if sample_bits:
                print(f"    sample: {', '.join(sample_bits)}")


def parse_datasets(values: Iterable[str]) -> list[str]:
    return [canonical_dataset_name(value) for value in values]


def target_dataset_name(dataset_name: str, embguardtest_target_name: Optional[str]) -> str:
    if dataset_name.startswith("EMBGuardTest") and embguardtest_target_name:
        return embguardtest_target_name
    return dataset_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate EMBGuard HF datasets from Hub to Hub.")
    parser.add_argument("--source-org", "--org", dest="source_org", default="EMBGuard")
    parser.add_argument("--target-org", default=None, help="Target org. Defaults to --source-org.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["EMBGuardTest", "heldout_set", "EMBHazard"],
        help="Datasets to migrate: EMBGuardTest, EMBGuardTest_v2, heldout_set, EMBHazard",
    )
    parser.add_argument("--source-revision", default=None, help="Optional source dataset revision.")
    parser.add_argument(
        "--embguardtest-target-name",
        default=None,
        help="Target repo name for EMBGuardTest only, e.g. EMBGuardTest_v2. Other datasets keep their names.",
    )
    parser.add_argument("--private", action="store_true", help="Make target datasets private.")
    parser.add_argument("--dry-run", action="store_true", help="Build transformed datasets without pushing.")
    parser.add_argument("--token", default=None, help="HF token. Defaults to HF_TOKEN.")
    parser.add_argument(
        "--commit-message",
        default="Migrate EMBGuard type split names and labels",
        help="Commit message for push_to_hub.",
    )
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    target_org = args.target_org or args.source_org
    dataset_names = parse_datasets(args.datasets)

    if token and not args.dry_run:
        login(token=token)
    elif args.dry_run:
        print("Dry run enabled: push_to_hub will be skipped.")

    for dataset_name in dataset_names:
        source_repo = f"{args.source_org}/{dataset_name}"
        target_name = target_dataset_name(dataset_name, args.embguardtest_target_name)
        target_repo = f"{target_org}/{target_name}"
        source = _load_dataset_dict(source_repo, token=token, revision=args.source_revision)
        migrated = migrate_dataset(dataset_name, source)
        print_dataset_summary(target_name, migrated)

        if args.dry_run:
            print(f"Dry run: not pushing {target_repo}")
            continue

        print(f"\nPushing {dataset_name} to {target_repo}")
        migrated.push_to_hub(
            target_repo,
            private=args.private,
            token=token,
            commit_message=args.commit_message,
        )
        print(f"Uploaded: https://huggingface.co/datasets/{target_repo}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
