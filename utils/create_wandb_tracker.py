import wandb 

def create_wandb_tracker(wandb_project_name, 
                        wandb_entity, 
                        config_dict,
                        wandb_group=None,
                        wandb_job_type=None,
                        wandb_name=None,):
    tracker = wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        config=config_dict,
        group=wandb_group,
        job_type=wandb_job_type,
        name=wandb_name,
    ) 

    return tracker