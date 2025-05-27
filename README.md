# LangChainDemo
Learning LangChain

## All about venvs

#### Create venv:
python -m venv myenv

#### Activate venv:
myenv\Scripts\activate

#### Create venv of specific python version:
provided python.exe for that version is renamed to pythonVersion.exe (Eg. python39.exe)
python39 -m venv myenv

#### Delete venv:
rmdir /s /q myenv

#### To add venvs as a kernel (myenv39 in activate mode):
pip install kernel
python -m ipykernel install --user --name=myenv39 --display-name "Python 39"

#### If kernel keeps disconnecting:
pip install --upgrade ipykernel

#### In vscode terminal:
lstmenv\Scripts\python311.exe -m ipykernel install --user --name lstmenv --display-name "Python 3.11 (lstmenv)"

