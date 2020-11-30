from unittest.mock import patch

import wandb


def call_wandb_api():
    return wandb.Api().run("Salman")


with patch.object(wandb.Api, 'run', return_value=2) as mock_method:
    thing = wandb.Api()
    print(thing.run(1, 2, 3))

mock_method.assert_called_once_with(1, 2, 3)

# with patch("wandb.Api") as wandb_api:
#     wandb_api.run.return_value = 2
#     r = call_wandb_api()

#     salman = 2
