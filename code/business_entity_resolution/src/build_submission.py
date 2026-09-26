import os
import zipfile
import argparse

def create_submission_zip(output_dir="output", zip_path="submission.zip"):
    results_file = os.path.join(output_dir, "matching_results.tsv")
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run the pipeline first.")
        return
        
    print(f"Creating submission zip: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        print(f"  Adding {results_file}...")
        zipf.write(results_file, arcname="matching_results.tsv")
        
        # Include code if requested by standard competition rules
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    print(f"  Adding {filepath}...")
                    zipf.write(filepath, arcname=os.path.join("code", file))
                    
    print(f"\nDone! Submission saved to {zip_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build submission zip file")
    parser.add_argument("--output_dir", type=str, default="../output")
    parser.add_argument("--zip_name", type=str, default="../submission.zip")
    args = parser.parse_args()
    create_submission_zip(args.output_dir, args.zip_name)
