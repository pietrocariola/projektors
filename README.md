Clone the repository:
```bash
git clone https://github.com/pietrocariola/projektors.git
cd projektors
```

Create a virtual environment:

```bash
conda create -n projektors
conda activate projektors
pip install -r requirements.txt
```

Or:

```bash
python -m venv projektors
source projektors/bin/activate
pip install -r requirements.txt
```

Download the datasets from their original sources: [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset), [Dermnet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet).

Go into ```/projektors/datasets/agri.py``` and ```/projektors/datasets/derm.py``` and change the constant ```DS_PATH``` to the places where the datasets are located in your computer.

Go one level above ```/projektors```:
```bash
cd /your/path/to/projektors
cd ..
```

In order to train the auxiliary models run:
```bash
./projektors/train_models.sh
```

To evaluate the representations before and after the projector run:
```bash
./projektors/eval_models.sh
```

Go inside every directory ```model-dataset/mlp``` and ```model-dataset/transformer``` and run the ```results.ipynb``` notebook to see the results. Example ```/projektors/llava-agri/mlp/results.ipynb```

Final tables can be found in : ```/projektors/results.xlsx```

Datasets splits can be found in: ```/projektors/datasets```

Model architectures can be found in: ```/projektors/models```

