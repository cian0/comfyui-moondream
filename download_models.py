from huggingface_hub import snapshot_download,hf_hub_download
import os
# local_dir=os.path.join(__file__,'./checkpoints')


current_file_dir = os.path.dirname(os.path.realpath(__file__))
# Correctly join the directory with the relative checkpoints path
local_dir = os.path.join(current_file_dir, 'checkpoints')
print(os.path.exists(local_dir))


print(os.path.exists(local_dir))
# snapshot_download("vikhyatk/moondream1",
#                                local_dir=local_dir,
#                                endpoint='https://hf-mirror.com')
print(local_dir)


hf_hub_download("vikhyatk/moondream1",local_dir=local_dir,
                                                filename="config.json",
                                                endpoint='https://hf-mirror.com')
                
hf_hub_download("vikhyatk/moondream1",local_dir=local_dir,
                                               filename="model.safetensors",
                                               endpoint='https://hf-mirror.com')
                
hf_hub_download("vikhyatk/moondream1",local_dir=local_dir,
                                               filename="tokenizer.json",
                                               endpoint='https://hf-mirror.com')
                    