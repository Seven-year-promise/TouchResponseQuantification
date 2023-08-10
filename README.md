## Quantification Platform for Touch Response of Zebrafish Larvae using Machine Learning

- Paper link: https://library.oapen.org/bitstream/handle/20.500.12657/52053/9783731511311.pdf?sequence=1#page=49
§split vides:

## The functions of this code

- Quantification of the data collected by [Touch-Response data acquisition platforms](https://github.com/Seven-year-promise/MultiFishTouchResponse);
- Visualization of the quantification results.

## This code is implemented by Python, and uses the following (parts) libraries:

- tensorflow-gpu==1.15.0
- torch=1.4.0
- scikit-learn==0.23.1
- scipy==1.6.0
- seaborn==0.12.2
- pyqt5-sip==4.19.18

## How to use the repostory

- Install the environment according to `environment.yml`.
- Change the path of the data and the path to save the quantification results in `config.py`: `QUANTIFY_DATA_PATH`, and `QUANTIFY_SAVE_PATH`.
- DO the quantification by
    `python quantification.py`
- Visulization of the results can be done by `./QuantificationResults/FigureDraw.py`


## In case of citing our work

```
@inproceedings{wang2021quantification,
  title={Quantification Platform for Touch Response of Zebrafish Larvae using Machine Learning},
  author={Wang, Yanke and Pylatiuk, Christian and Mikut, Ralf and Peravali, Ravindra and Reischl, Markus},
  booktitle={PROCEEDINGS 31. WORKSHOP COMPUTATIONAL INTELLIGENCE},
  volume={25},
  pages={37},
  year={2021}
}
```
