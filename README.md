# DPT (**Dual Modality Prompt Tuning for Vision Language Models**)

This is a Course Project done as part of the course IE-643 which involves prompt tuning for both vision and language modes and improve the performace of those models on downstream specific tasks


## Installation Instructions
Make a virtual environment
```
python3 -m venv <virt-env-name>
```
Then install all the dependencies
```
pip install -r requirements.txt
```

## Testing Instructions
- Currently it is set for Medical Dataset
- Put the image path in `train_single.sh` within the `--heatmap-impath` flag and the heapmap along with the original image will be shown by running the following command
```
./train_single.sh 0
```
where `0` is the **random seed**


