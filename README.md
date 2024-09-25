# A Feature Generator for Few-Shot Learning

Train the generator using `traingerator.py` 

## Testing 
Set the path of the trained generator in `modified-few-shot-meta-baseline/models/meta_baseline.py`

For testing its effect on the MetaBaseline model. We'll be using a modified version of MetaBaseline, located in the `modified-few-shot-meta-baseline` folder.  refer to the original [MetaBaseline GitHub repository](https://github.com/yinboc/few-shot-meta-baseline) for more details on the evaluation process.

## Prerequisites

Before starting, ensure you have installed the necessary dependencies for both the generator and MetaBaseline:

1. **Check `requirements.txt` for Dependencies**:  
   Install the required dependencies by running the following:

   ```bash
   pip install -r requirements.txt
