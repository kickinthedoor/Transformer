def inspect_dataset_samples(
    src_encoded: list[list[int]],
    tgt_encoded: list[list[int]],
    src_tokenizer,
    tgt_tokenizer,
    num_samples: int = 3,
    start_idx: int = 0
):
    """
    Prints parallel source and target samples showing integer token IDs,
    subword piece breakdown, and the reconstructed plain text.
    """
    print("=" * 80)
    print(f" DATASET INSPECTION ({num_samples} samples starting from index {start_idx})")
    print("=" * 80)

    for i in range(start_idx, start_idx + num_samples):
        src_ids = src_encoded[i]
        tgt_ids = tgt_encoded[i]

        # 1. Decode back to full text strings
        src_text = src_tokenizer.decode(src_ids)
        tgt_text = tgt_tokenizer.decode(tgt_ids)

        # 2. Extract subword pieces to see how words were split
        src_pieces = [src_tokenizer.processor.id_to_piece(idx) for idx in src_ids]
        tgt_pieces = [tgt_tokenizer.processor.id_to_piece(idx) for idx in tgt_ids]

        print(f"\n--- [ Sample #{i + 1} ] ---")

        # SOURCE (German)
        print("\n🇩🇪 SOURCE (German):")
        print(f"  • Plain Text: {src_text}")
        print(f"  • Token IDs ({len(src_ids)}): {src_ids}")
        print(f"  • Subword Pieces: {src_pieces}")

        # TARGET (English)
        print("\n🇬🇧 TARGET (English):")
        print(f"  • Plain Text: {tgt_text}")
        print(f"  • Token IDs ({len(tgt_ids)}): {tgt_ids}")
        print(f"  • Subword Pieces: {tgt_pieces}")

    print("\n" + "=" * 80)
