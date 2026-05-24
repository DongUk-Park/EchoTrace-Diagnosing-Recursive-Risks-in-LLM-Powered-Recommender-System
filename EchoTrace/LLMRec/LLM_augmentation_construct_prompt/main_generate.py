from LLM_augmentation_construct_prompt import gpt_user_profiling_parallel
from LLM_augmentation_construct_prompt import gpt_ui_aug_parallel
from LLM_augmentation_construct_prompt import gpt_i_attribute_generate_aug
import pickle

def main(dataset, file_path=None, profile_step_num=None, parts=None):
    print("Running gpt_user_profiling...")
    gpt_user_profiling_parallel.main(
        dataset,
        provider="anthropic",#
        file_path=file_path,
        profile_step_num=profile_step_num,
        parts=parts,
    )  # 또는 provider="anthropic"
    
    print("Running gpt_ui_aug...")
    gpt_ui_aug_parallel.main(dataset, file_path=file_path, provider="anthropic")   
    
    # print("Running gpt_i_attribute_generate_aug...")
    # gpt_i_attribute_generate_aug.main(dataset) # only 1 time
    
if __name__ == "__main__":
    main("yelp")
