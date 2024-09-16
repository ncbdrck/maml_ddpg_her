# maml_ddpg_her

Download the code from the repository:
```bash
git clone https://github.com/ncbdrck/maml_ddpg_her.git
```

Install the required packages:
```bash
cd maml_ddpg_her
pip install -r requirements.txt
```

run the code:
```bash
cd maml_ddpg_her
mpiexec -np 8 python3 train.py --cuda

# or
mpirun -np 8 python3 train.py --cuda
```