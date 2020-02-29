This repository contains a Python reimplementation of "Bayesian Personalized Ranking for Novelty Enhancement" [Wasilewski and Hurley, UMAP'19].

## Visualize Psudo Data
```bash
MODEL_NAME=novelty_enhanced_bpr python main.py novelty_enhanced_bpr.VisualizePsudoData --local-scheduler
```

## Training
```bash
MODEL_NAME=novelty_enhanced_bpr python main.py novelty_enhanced_bpr.TrainModel --local-scheduler
```

## Test
```bash
MODEL_NAME=novelty_enhanced_bpr python main.py novelty_enhanced_bpr.TestModel --local-scheduler
```
