Clone the repository:
```bash
git clone https://github.com/pietrocariola/projektors.git
```

Download the datasets from their original sources: [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset), [Dermnet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet).

Go into ```/projektors/datasets/agri.py``` and ```/projektors/datasets/derm.py``` and change the constant ```DS_PATH``` to the places where the datasets are located in your computer.

Go one level above ```/projektors```:
```bash
cd /your/path/to/projektors
cd ..
```

Run ```agri.py``` and ```derm.py``` to create the datasets' jsons:
```bash
python -m projektors.datasets.agri
python -m projektors.datasets.derm
```

In order to train the auxiliary models run:
```bash
./projektors/train_models.sh
```

To evaluate the representations before and after the projector run:
```bash
./projektors/eval_models.sh
```



