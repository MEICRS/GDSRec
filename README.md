
A PyTorch implementation of GDSRec

1. Install required packages from requirements.txt file.
```bash
pip install -r requirements.txt
```

2. Preprocess dataset. Two pkl files named dataset and list should be generated in the respective folders of the dataset.
```bash
python preprocess.py --dataset Ciao
python preprocess.py --dataset Epinions
```

3. Run main.py file to train the model. You can configure some training parameters through the command line. 
```bash
python main.py
```

4. Run test.py file to test the model.
```bash
python test.py
```

For the ranking task, please see https://github.com/MEICRS/GDSRec_rank

