import argparse
from pathlib import Path
import pandas as pd
from pdb import set_trace

from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber


def split_dataframe(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42):
    """Shuffle and split a DataFrame into train/val/test parts."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def save_split(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_feather(path)
    return path


def process_split(raw_path: Path, out_dir: Path, prober: IlluminaMethylationProber, embedder: DNALLMEmbedder) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # transform the raw probe data into memory-mapped data
    print(f"Processing raw data from {raw_path} and saving to {out_dir}")
    saver = CpGPTDataSaver(data_paths=str(raw_path), processed_dir=str(out_dir), metadata_cols=["haschip"])
    saver.process_files(prober, embedder)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CpGPT dataset with haschip target")
    parser.add_argument("--methylation_data", default="/grand/GeomicVar/tarak/cpgpt/CpGPT/data/tutorials/raw/fhs_filtered.arrow", help="Input methylation data file (arrow or feather)")
    parser.add_argument("--metadata_file", default="/grand/GeomicVar/tarak/methylGPT/data_kirmani/fhs_chip_metadata_yp_05092025.tsv", help="Metadata file containing haschip target variable")
    parser.add_argument("--data_dir", default="../data/haschip", help="Output dataset directory")
    parser.add_argument("--dependencies", default="/grand/GeomicVar/tarak/cpgpt/CpGPT/dependencies/human", help="DNA LLM dependencies directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    
    # Read methylation data
    print("Reading methylation data from:", args.methylation_data)
    methylation_df = pd.read_feather(args.methylation_data)
    print("Methylation data shape:", methylation_df.shape)
    # print("Methylation data columns:", methylation_df.columns.tolist())
    
    # Read metadata file
    print("Reading metadata from:", args.metadata_file)
    metadata_df = pd.read_csv(args.metadata_file, sep="\t")
    print("Metadata shape:", metadata_df.shape)
    print("Metadata columns:", metadata_df.columns.tolist())
    
    # # Check if Sample column exists in metadata
    # if 'Sample' not in metadata_df.columns:
    #     print("ERROR: 'Sample' column not found in metadata!")
    #     print("Available columns in metadata:", metadata_df.columns.tolist())
    #     return
    
    # Check if subject_id exists in metadata for merging
    if 'subject_id' not in metadata_df.columns:
        print("ERROR: 'subject_id' column not found in metadata!")
        print("Available columns in metadata:", metadata_df.columns.tolist())
        return
    
    # Check if sample_id exists as index in methylation data
    if methylation_df.index.name != 'sample_id':
        print("WARNING: methylation data index name is not 'sample_id'")
        print("Index name:", methylation_df.index.name)
        print("First few index values:", methylation_df.index[:5].tolist())
    
    # Reset index to make sample_id a column for merging
    print("Resetting methylation data index to make sample_id a column...")
    methylation_df = methylation_df.reset_index()
    print("Methylation data columns after reset:", methylation_df.columns.tolist())
    
    # Check for repeated entries in both dataframes
    print("Checking for repeated entries...")
    methylation_duplicates = methylation_df['sample_id'].duplicated().sum()
    metadata_duplicates = metadata_df['subject_id'].duplicated().sum()
    print(f"Repeated sample_ids in methylation data: {methylation_duplicates}")
    print(f"Repeated subject_ids in metadata: {metadata_duplicates}")
    
    # Check data types for merging
    print("Data type check for merging:")
    print(f"sample_id dtype: {methylation_df['sample_id'].dtype}")
    print(f"subject_id dtype: {metadata_df['subject_id'].dtype}")
    print(f"First few sample_ids: {methylation_df['sample_id'].head().tolist()}")
    print(f"First few subject_ids: {metadata_df['subject_id'].head().tolist()}")
    
    # Convert both to string for consistent merging
    print("Converting both IDs to string type for consistent merging...")
    methylation_df['sample_id'] = methylation_df['sample_id'].astype(str)
    metadata_df['subject_id'] = metadata_df['subject_id'].astype(str)
    print(f"After conversion - sample_id dtype: {methylation_df['sample_id'].dtype}")
    print(f"After conversion - subject_id dtype: {metadata_df['subject_id'].dtype}")

    # set_trace()
    
    if methylation_duplicates > 0:
        print("Removing duplicate sample_ids from methylation data...")
        methylation_df = methylation_df.drop_duplicates(subset=['sample_id'])
        print("Methylation data shape after removing duplicates:", methylation_df.shape)
    
    if metadata_duplicates > 0:
        print("Removing duplicate subject_ids from metadata...")
        metadata_df = metadata_df.drop_duplicates(subset=['subject_id'])
        print("Metadata shape after removing duplicates:", metadata_df.shape)
    
    # Check haschip column in metadata
    if 'haschip' not in metadata_df.columns:
        print("ERROR: 'haschip' column not found in metadata!")
        print("Available columns in metadata:", metadata_df.columns.tolist())
        return
    
    print("haschip column found in metadata!")
    print("haschip dtype:", metadata_df['haschip'].dtype)
    print("haschip unique values:", metadata_df['haschip'].unique())
    print("haschip value counts:")
    print(metadata_df['haschip'].value_counts())
    
    # Convert haschip to numeric if needed
    if metadata_df['haschip'].dtype == 'object':
        print("Converting haschip from object to numeric...")
        metadata_df['haschip'] = pd.to_numeric(metadata_df['haschip'], errors='coerce')
        print("After conversion - haschip dtype:", metadata_df['haschip'].dtype)
        print("haschip unique values:", metadata_df['haschip'].unique())
    
    # Ensure it's int for binary classification
    metadata_df['haschip'] = metadata_df['haschip'].astype(int)
    print("Final haschip dtype:", metadata_df['haschip'].dtype)
    
    # Merge methylation data with metadata
    print("Merging methylation data with metadata...")
    print("Methylation data samples:", len(methylation_df))
    print("Metadata samples:", len(metadata_df))
    
    # Merge on sample_id (methylation) and subject_id (metadata)
    merged_df = methylation_df.merge(metadata_df[['subject_id', 'haschip']], left_on='sample_id', right_on='subject_id', how='inner')
    # set_trace()
    print("Merged data shape:", merged_df.shape)
    print("Merged data samples:", len(merged_df))
    
    # Check for missing haschip values
    missing_haschip = merged_df['haschip'].isna().sum()
    print(f"Missing haschip values: {missing_haschip}")
    
    if missing_haschip > 0:
        print("Removing rows with missing haschip values...")
        merged_df = merged_df.dropna(subset=['haschip'])
        print("After removing missing values, shape:", merged_df.shape)
    
    # Verify haschip column is present and properly formatted
    if 'haschip' not in merged_df.columns:
        print("ERROR: haschip column not found after merge!")
        return
    
    print("Final haschip statistics:")
    print("haschip dtype:", merged_df['haschip'].dtype)
    print("haschip unique values:", merged_df['haschip'].unique())
    print("haschip value counts:")
    print(merged_df['haschip'].value_counts())

    print("Splitting data into train, val, and test sets...")
    train_df, val_df, test_df = split_dataframe(merged_df)

    raw_base = data_dir / "raw"
    print(f"Saving splits to {raw_base}")
    print("Saving train split:", train_df.shape)
    train_file = save_split(train_df, raw_base / "train.arrow")
    print("Saving val split:", val_df.shape)
    val_file = save_split(val_df, raw_base / "val.arrow")
    print("Saving test split:", test_df.shape)
    test_file = save_split(test_df, raw_base / "test.arrow")
    print("Train, val, and test splits saved successfully.")

    # Memory-map Data
    # In order to perform inference, we need to memory-map the data. 
    # This is done by using the CpGPTDataSaver class.
    # We first need to define the DNALLMEmbedder and IlluminaMethylationProber classes, 
    # which contain the information about the DNA LLM Embeddings and the conversion between Illumina array probes to genomic locations, respectively.

    # Use CpGPT's built-in DNALLMEmbedder and IlluminaMethylationProber to convert raw data into: i) DNA‑LM embeddings (sequence context), and 2) CpG site probe vectors (as used during training)

    # Reads DNA-LM weights and vocabulary files from args.dependencies; When later applied to a CpG site, it retrieves the DNA sequence window (e.g., ±k bases) and converts that into a numeric embedding vector using the transformer model.
    print("********** Initializing DNALLMEmbedder...")
    embedder = DNALLMEmbedder(dependencies_dir=args.dependencies)

    # Translates Illumina array CpG probe IDs into genomic context and methylation signals
    # Loads array annotation files (probe IDs, chromosome, coordinates) from args.dependencies.
    # Uses the embedder to pull DNA sequence embeddings around each probe's genomic coordinate.
    # Converts raw methylation values (e.g., beta values) from  input into standardized model inputs
    print("********** Initializing IlluminaMethylationProber...")
    prober = IlluminaMethylationProber(dependencies_dir=args.dependencies, embedder=embedder)
    # set_trace()
    # Process the splits and save them in the processed directory
    print(f"Processing train split: {train_file}")
    process_split(train_file, data_dir / "processed" / "train", prober, embedder)
    print(f"Processing val split: {val_file}")
    process_split(val_file, data_dir / "processed" / "val", prober, embedder)
    print(f"Processing test split: {test_file}")
    process_split(test_file, data_dir / "processed" / "test", prober, embedder)


if __name__ == "__main__":
    main()