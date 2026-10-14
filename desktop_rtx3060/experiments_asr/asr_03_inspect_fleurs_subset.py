from experiments_asr.asr_01_fleurs_subset import load_audio_from_entry, load_manifest, load_metadata, validate_manifest


def main() -> None:
    metadata = load_metadata()
    entries = load_manifest()

    validate_manifest(
        entries,
        metadata,
    )

    print("=" * 70)
    print("FLEURS SUBSET VALIDATION")
    print("=" * 70)

    print(f"Manifest samples : {len(entries)}")
    print(f"SHA256           : {metadata['manifest_sha256']}")

    languages = metadata["languages"].keys()

    for language in languages:
        entry = next(item for item in entries if item["language"] == language)

        samples, sample_rate, reference = load_audio_from_entry(
            entry,
            metadata,
        )

        duration = len(samples) / sample_rate

        print()
        print("-" * 70)
        print(f"{language.upper()} ({entry['dataset_config']})")
        print("-" * 70)

        print(f"Dataset index : {entry['dataset_index']}")
        print(f"Sample ID     : {entry['sample_id']}")
        print(f"Sample rate   : {sample_rate} Hz")
        print(f"Samples       : {len(samples):,}")
        print(f"Duration      : {duration:.3f} s")
        print(f"Dtype         : {samples.dtype}")
        print(f"Shape         : {samples.shape}")
        print(f"Reference     : {reference}")

    print()
    print("=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
