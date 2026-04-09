import argparse
from megatron.energon import (
    VQASample,
    WorkerConfig,
    get_loader,
    get_train_dataset,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    worker_config = WorkerConfig.default_worker_config(0)

    train_ds = get_train_dataset(
        args.data_path,
        batch_size=1,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        worker_config=worker_config,
    )

    loader = get_loader(train_ds, worker_config=worker_config)

    stats = {
        "total": 0,
        "has_image": 0,
        "image_tag_in_context": 0,
        "image_tag_in_first_turn": 0,
        "image_tag_in_later_turn": 0,
        "multi_turn": 0,
    }

    for i, sample in enumerate(loader):
        if i >= args.num_samples:
            break
        stats["total"] += 1

        if isinstance(sample, list):
            sample = sample[0]
        if not isinstance(sample, VQASample):
            print(f"--- Sample {i}: not VQASample, type={type(sample)} ---")
            continue

        # Normalize context and answers to lists
        contexts = sample.context if isinstance(sample.context, list) else [sample.context]
        answers = sample.answers if isinstance(sample.answers, list) else [sample.answers]

        num_turns = len(contexts)
        if num_turns > 1:
            stats["multi_turn"] += 1

        has_image = sample.image is not None
        if has_image:
            stats["has_image"] += 1

        print(f"--- Sample {i} ---")
        print(f"  num_turns: {num_turns}")
        print(f"  has_image: {has_image}")
        if has_image:
            print(f"  image shape: {sample.image.shape}, dtype: {sample.image.dtype}")

        # Check each turn for <image> tag
        found_image_tag = False
        for turn_idx, ctx in enumerate(contexts):
            has_tag = "<image>" in ctx
            ans = answers[turn_idx] if turn_idx < len(answers) else "(no answer)"

            print(f"  [Turn {turn_idx}]")
            print(f"    context:       {repr(ctx[:200])}")
            print(f"    answer:        {repr(ans[:200]) if isinstance(ans, str) else ans}")
            print(f"    has <image>:   {has_tag}")

            if has_tag:
                found_image_tag = True
                if turn_idx == 0:
                    stats["image_tag_in_first_turn"] += 1
                else:
                    stats["image_tag_in_later_turn"] += 1

        if found_image_tag:
            stats["image_tag_in_context"] += 1

        # Warn if there's an image but no <image> tag anywhere
        if has_image and not found_image_tag:
            print(f"  ⚠️  WARNING: sample has image but NO <image> tag in any turn!")

        print()

    print("=" * 50)
    print("Summary:")
    print(f"  Total samples checked:        {stats['total']}")
    print(f"  Samples with image:           {stats['has_image']}")
    print(f"  Samples with <image> in text: {stats['image_tag_in_context']}")
    print(f"  <image> in first turn:        {stats['image_tag_in_first_turn']}")
    print(f"  <image> in later turn:        {stats['image_tag_in_later_turn']}")
    print(f"  Multi-turn samples:           {stats['multi_turn']}")

if __name__ == "__main__":
    main()