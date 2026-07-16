# качает сабсэмпл VK-LSVD и метаданные с HuggingFace
import argparse

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "deepvk/VK-LSVD"
LOCAL_DIR = "data/VK-LSVD"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", default="ur0.01_ir0.01")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="skip the 2.4 GB item_embeddings.npz")
    args = parser.parse_args()

    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    wanted = [f for f in all_files
              if f.startswith(f"subsamples/{args.subsample}/")
              or f.startswith("metadata/")]
    if args.skip_embeddings:
        wanted = [f for f in wanted if not f.endswith(".npz")]

    print(f"Downloading {len(wanted)} files to {LOCAL_DIR}")
    for f in sorted(wanted):
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                        filename=f, local_dir=LOCAL_DIR)
        print(f"  ok: {f}")
    print("Done.")


if __name__ == "__main__":
    main()
