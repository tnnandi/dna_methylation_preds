# Standard library imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
import json
from pdb import set_trace


warnings.simplefilter(action="ignore", category=FutureWarning)

# Data science imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaging as pya
import seaborn as sns

# Lightning imports
from lightning.pytorch import seed_everything


# cpgpt-specific imports
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.model.cpgpt_module import m_to_beta

def main():
    plot_stats = True  # set to True to plot statistics for metadata columns
    # MODEL_NAME = "age"
    MODEL_NAME = "clock_proxies"
    ARROW_DF_FILTERED_PATH = "../data/tutorials/raw/fhs_filtered.arrow"
    MAX_INPUT_LENGTH = 200_000 #40_000
    MAX_ATTN_LENGTH = 1_000

    # Set constants
    RANDOM_SEED = 42
    DEPENDENCIES_DIR = "../dependencies"
    LLM_DEPENDENCIES_DIR = DEPENDENCIES_DIR + "/human"
    DATA_DIR = "../data"
    PROCESSED_DIR = "../data/tutorials/processed/fhs_setup"
    
    # MODEL_NAME = "age"
    # MODEL_NAME = "clock_proxies"
    MODEL_CHECKPOINT_PATH = f"../dependencies/model/weights/{MODEL_NAME}.ckpt"
    MODEL_CONFIG_PATH = f"../dependencies/model/config/{MODEL_NAME}.yaml"
    MODEL_VOCAB_PATH = f"../dependencies/model/vocab/{MODEL_NAME}.json"
    
    # ARROW_DF_FILTERED_PATH = "../data/tutorials/raw/fhs_filtered.arrow"
    # MAX_INPUT_LENGTH = 20_000
    # MAX_ATTN_LENGTH = 1_000

    # Set random seed
    seed_everything(RANDOM_SEED, workers=True)

    # Initialize inferencer
    inferencer = CpGPTInferencer(dependencies_dir=DEPENDENCIES_DIR, data_dir=DATA_DIR, offline=False)

    inferencer.download_dependencies(species="human")

    # To generate genomic embeddings for loci outside of the ones already available for download
    if not os.path.exists(LLM_DEPENDENCIES_DIR):
        # List CpG genomic locations
        example_genomic_locations = ['1:100000', '1:250500', 'X:2031253'] #edit it to include the genomic locations you want to embed

        # Declare required class
        embedder = DNALLMEmbedder(dependencies_dir=LLM_DEPENDENCIES_DIR)

        # Parse the embeddings
        embedder.parse_dna_embeddings(
            example_genomic_locations,
            "homo_sapiens",
            dna_llm="nucleotide-transformer-v2-500m-multi-species",
            dna_context_len=2001,
        )
    
    # Load FHS data
    root_dir = "/grand/GeomicVar/tarak/cpgpt/CpGPT/data_kirmani"
    data_dir = os.path.join(root_dir, "phg001091.v5.FHS_DNAMethylation.methylation-data-matrixfmt.c1")
    gen3_parquet = os.path.join(data_dir, "gen3_methylation_c1.parquet")
    umn_parquet = os.path.join(data_dir, "UMN_methylation_c1.parquet")
    jhu_parquet = os.path.join(data_dir, "JHU_methylation_c1.parquet")
    
    # Load data and metadata
    # Read each Parquet file into a DataFrame
    df_gen3 = pd.read_parquet(gen3_parquet)
    df_umn = pd.read_parquet(umn_parquet)
    df_jhu = pd.read_parquet(jhu_parquet)

    # Concatenate all dataframes 
    df = pd.concat([df_gen3, df_umn, df_jhu], axis=1) # total samples = 3847
    print(f"Total samples after concatenation: {df.shape[1]}")

    metadata_df = pd.read_csv("/grand/GeomicVar/tarak/methylGPT/data_kirmani/fhs_chip_metadata_yp_05092025.tsv", sep="\t")
    
    # Process dataframes
    df = df.T
    print(f"Total samples after transpose: {len(df)}")
    df.index.name = 'sample_id'
    df.columns.name = None
    # set_trace()
    
    # Filter and merge data
    metadata_df['subject_id'] = metadata_df['subject_id'].astype(str)
    common_ids = set(df.index.astype(str)).intersection(set(metadata_df['subject_id'].astype(str)))

    # Filter both dataframes based on common IDs
    filtered_df = df[df.index.astype(str).isin(common_ids)]
    print(f"Total samples after ID filtering: {len(filtered_df)}")
    filtered_metadata = metadata_df[metadata_df['subject_id'].astype(str).isin(common_ids)]
    filtered_metadata = filtered_metadata.drop_duplicates(subset='subject_id')
    # Ensure the order of IDs is consistent
    common_ids_ordered = [id_ for id_ in df.index.astype(str) if id_ in common_ids]
    # Reorder both dataframes based on common IDs
    filtered_df = filtered_df.loc[common_ids_ordered]
    filtered_metadata = (
        filtered_metadata
        .set_index('subject_id')
        .loc[common_ids_ordered]
        .reset_index()
    )
    # set_trace()

    if plot_stats:
        # create histograms for AgeAtBloodDraw, sex, haschip, gene, ExonicFunc, VAF
        # Create a figure with subplots
        plt.figure(figsize=(15, 10))

        # Age histogram
        plt.subplot(2, 3, 1)
        sns.histplot(data=filtered_metadata, x='AgeAtBloodDraw', bins=30)
        plt.title('Age Distribution')

        # Sex distribution
        plt.subplot(2, 3, 2)
        sns.countplot(data=filtered_metadata, x='sex')
        plt.title('Sex Distribution')

        # Haschip distribution
        plt.subplot(2, 3, 3)
        sns.countplot(data=filtered_metadata, x='haschip')
        plt.title('Has CHIP Distribution')

        # Gene distribution 
        plt.subplot(2, 3, 4)
        sns.countplot(data=filtered_metadata, x='Gene')
        plt.xticks(rotation=45, ha='right')
        plt.title('Gene Distribution')

        # ExonicFunc distribution
        plt.subplot(2, 3, 5)
        sns.countplot(data=filtered_metadata, x='ExonicFunc')
        plt.xticks(rotation=45, ha='right')
        plt.title('Exonic Function Distribution')

        # VAF histogram
        plt.subplot(2, 3, 6)
        sns.histplot(data=filtered_metadata, x='VAF', bins=30)
        plt.title('VAF Distribution')

        plt.tight_layout()
        plt.savefig('metadata_distributions.png')
        plt.close()

    # set_trace()
    
    # Download and load model
    model_exists = (
    os.path.exists(MODEL_CHECKPOINT_PATH) and
    os.path.exists(MODEL_CONFIG_PATH) and
    os.path.exists(MODEL_VOCAB_PATH)
    )

    if not model_exists:
        inferencer.download_model(MODEL_NAME)
    else:
        print("Model files found locally, skipping download_model()")

    # inferencer.download_model(MODEL_NAME)
    config = inferencer.load_cpgpt_config(MODEL_CONFIG_PATH)
    model = inferencer.load_cpgpt_model(config, model_ckpt_path=MODEL_CHECKPOINT_PATH, strict_load=True)
    
    # Filter vocab features
    # filtering its columns to include only those CpG sites or probes that are present in the model's vocabulary file
    vocab = json.load(open(MODEL_VOCAB_PATH, 'r'))
    filtered_df = filtered_df.loc[:, filtered_df.columns.isin(vocab['input'])]
    print(f"Total features after vocab filtering (keeping only probes that are common to those used in model training): {filtered_df.shape[1]}")
    filtered_df.to_feather(ARROW_DF_FILTERED_PATH)
    # set_trace()
    # Setup data processing
    embedder = DNALLMEmbedder(dependencies_dir=LLM_DEPENDENCIES_DIR)
    prober = IlluminaMethylationProber(dependencies_dir=LLM_DEPENDENCIES_DIR, embedder=embedder)
    
    # Process data
    print(f"Samples being processed: {filtered_df.shape[0]}")
    quick_setup_datasaver = CpGPTDataSaver(data_paths=ARROW_DF_FILTERED_PATH, processed_dir=PROCESSED_DIR)
    quick_setup_datasaver.process_files(prober, embedder)
    
    # Setup data modules
    quick_setup_datamodule = CpGPTDataModule(
        predict_dir=PROCESSED_DIR,
        dependencies_dir=LLM_DEPENDENCIES_DIR,
        batch_size=128,
        num_workers=0, #255
        max_length=MAX_INPUT_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False,
    )
    
    quick_setup_datamodule_attn = CpGPTDataModule(
        predict_dir=PROCESSED_DIR,
        dependencies_dir=LLM_DEPENDENCIES_DIR,
        batch_size=8,
        num_workers=0,
        max_length=MAX_ATTN_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False,
    )
    
    # Initialize trainer
    trainer = CpGPTTrainer(precision="16-mixed")
    # set_trace()
    quick_setup_datamodule.setup(stage='predict')
    dataloader = quick_setup_datamodule.predict_dataloader()
    total_samples = len(dataloader.dataset)
    print(f"Total samples in dataloader: {total_samples}")
    # set_trace()

    # Get predictions
    sample_embeddings = trainer.predict(
        model=model,
        datamodule=quick_setup_datamodule,
        predict_mode="forward",
        # return_keys=["sample_embedding"]
    )
    
    # Save embeddings
    embeddings_dir = os.path.join(PROCESSED_DIR, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Save as numpy array
    embeddings_path = os.path.join(embeddings_dir, f'sample_embeddings_{MODEL_NAME}.npy')
    np.save(embeddings_path, sample_embeddings['sample_embedding'].cpu().numpy())
    
    # # Optionally save sample IDs if needed
    sample_ids = list(filtered_df.index)
    # sample_ids_path = os.path.join(embeddings_dir, 'sample_ids.json')
    # with open(sample_ids_path, 'w') as f:
    #     json.dump(sample_ids, f)

        # Save sample IDs and metadata
    metadata_dict = {
        'sample_ids': list(filtered_df.index),
        'subject_id': list(filtered_metadata['subject_id']),
        'Sample': list(filtered_metadata['Sample']),
        'AgeAtBloodDraw': list(filtered_metadata['AgeAtBloodDraw']),
        'sex': list(filtered_metadata['sex']),
        'PC1': list(filtered_metadata['PC1'].astype(float)),
        'PC2': list(filtered_metadata['PC2'].astype(float)),
        'PC3': list(filtered_metadata['PC3'].astype(float)),
        'PC4': list(filtered_metadata['PC4'].astype(float)),
        'PC9': list(filtered_metadata['PC9'].astype(float)),
        'PC10': list(filtered_metadata['PC10'].astype(float)),
        'PC11': list(filtered_metadata['PC11'].astype(float)),
        'haschip': list(filtered_metadata['haschip']),
        'Gene': list(filtered_metadata['Gene']),
        'ExonicFunc': list(filtered_metadata['ExonicFunc']),
        'VAF': list(filtered_metadata['VAF'].astype(float)),
        'chip_binary': list(filtered_metadata['chip_binary'])
    }
    
    metadata_path = os.path.join(embeddings_dir, 'metadata_true.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    # set_trace()

    pred_conditions = trainer.predict(
        model=model,
        datamodule=quick_setup_datamodule,
        predict_mode="forward",
        # return_keys=["pred_conditions", "condition_names"]
    )
    # set_trace()
    # Save predictions
    # predictions_dir = os.path.join(PROCESSED_DIR, 'predictions')
    # os.makedirs(predictions_dir, exist_ok=True)
    
    # Convert predictions to numpy and save
    predictions = pred_conditions['pred_conditions'].cpu().numpy()
    # condition_names = pred_conditions['condition_names']
    
    # Save predictions and condition names
    np.save(os.path.join(embeddings_dir, f'predictions_{MODEL_NAME}{MAX_INPUT_LENGTH}_{MAX_ATTN_LENGTH}.npy'), predictions)
    # Save predictions and sample IDs together
    predictions_dict = {
        'sample_ids': [str(id_) for id_ in filtered_df.index],  # Ensure IDs are strings
        'predictions': predictions.tolist()  # Convert numpy array to list for JSON serialization
    }
    
    predictions_path = os.path.join(embeddings_dir, f'predictions_with_ids_{MODEL_NAME}_{MAX_INPUT_LENGTH}_{MAX_ATTN_LENGTH}.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions_dict, f, indent=2)

    # Write true and predicted conditions to a JSON file
    sample_ids = [str(id_) for id_ in filtered_df.index]  # Ensure IDs are strings
    predictions = pred_conditions['pred_conditions'].cpu().numpy()
    true = filtered_metadata['AgeAtBloodDraw'].astype(float).tolist()  # Convert to list
    with open(os.path.join(embeddings_dir, f'true_and_predicted_{MODEL_NAME}_{MAX_INPUT_LENGTH}_{MAX_ATTN_LENGTH}.json'), 'w') as f:
        json.dump({
            'sample_ids': sample_ids,
            'true_conditions': true,
            'predicted_conditions': predictions.tolist()  # Convert numpy array to list for JSON serialization
        }, f, indent=2)


    set_trace()
        
    # Create DataFrame with predictions
    pred_df = pd.DataFrame(
        predictions, 
        # columns=condition_names,
        index=sample_ids
    )
    # set_trace()
    # Reconstruct methylation
    probes = list(df.columns[0:100])
    genomic_locations = prober.locate_probes(probes, "homo_sapiens")
    
    # pred_meth = trainer.predict(
    #     model=model,
    #     datamodule=quick_setup_datamodule,
    #     predict_mode="reconstruct",
    #     genomic_locations=genomic_locations,
    #     species="homo_sapiens",
    #     return_keys=["pred_meth"],
    # )
    
    # pred_meth["pred_meth"] = m_to_beta(pred_meth["pred_meth"])
    
    # # Get attention weights
    # attn_weights = trainer.predict(
    #     model=model,
    #     datamodule=quick_setup_datamodule_attn,
    #     predict_mode="attention",
    #     aggregate_heads="mean",
    #     layer_index=-1,
    #     return_keys=["attention_weights", "chroms", "positions", "mask_na", "meth"],
    # )
    
    return sample_embeddings, pred_conditions, pred_df, filtered_metadata #, pred_meth, attn_weights

if __name__ == "__main__":
    # sample_embeddings, pred_conditions, pred_meth, attn_weights = main()
    sample_embeddings, pred_conditions, pred_df, filtered_metadata = main()
    set_trace()