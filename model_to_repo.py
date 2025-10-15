from huggingface_hub import HfApi
from utils import *
import sys


# load config
config = load_json(sys.argv[1])

checkpoint = config.get("checkpoint", ModelUtils.get_latest_checkpoint(config['dir']))
model_path = f"results/{config['dir']}/checkpoint-{checkpoint}"

repo_id = "trietbui/model-kvasir-vqa-x1"

api = HfApi()
api.create_repo(repo_id, repo_type="model", exist_ok=True)

api.upload_folder(
    repo_id = repo_id,
    folder_path = model_path,
)