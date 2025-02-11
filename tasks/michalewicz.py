import sys 
sys.path.append("../")
import torch
import math
from tasks.objective import Objective
from botorch.test_functions import Michalewicz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Michalewicz10(Objective):
    """Hartmann function optimization task from
    https://www.sfu.ca/~ssurjano/hart6.html,
    Designed as a minimization task so we multiply by -1
    to obtain a maximization task 
    Using BoTorch implementation: 
    https://botorch.org/v/0.1.2/api/_modules/botorch/test_functions/hartmann6.html
    """
    def __init__(
        self,
        **kwargs,
    ):
        self.neg_michalewicz10 = Michalewicz(negate=True, dim=10)

        super().__init__(
            dim=10,
            lb=0,
            ub=math.pi,
            **kwargs,
        )

    def f(self, x):
        x = x.to(device)
        self.num_calls += 1
        y = self.neg_michalewicz10(x)
        return y.item()


if __name__ == "__main__":
    obj = Michalewicz10()
    known_best_eval = 9.66
    print(f"Best possible Michalewicz10 eval: {known_best_eval}")