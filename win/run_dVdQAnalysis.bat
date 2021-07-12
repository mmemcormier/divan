set myPATH=%UserProfile%\Anaconda3\Scripts
call %myPATH%\activate.bat
call conda create -n dVdQAnalysis python==3.7 /Q
call conda activate dVdQAnalysis

call pip install streamlit==0.81.1
call pip install scipy==1.5.2
call pip install matplotlib
pip install tk
pip install bokeh==2.0.2
conda install selenium gecko driver -c conda-forge
conda install selenium python-chromedriver-binary -c conda-forge


streamlit run dVdQAnalysis_Windows.py

call conda deactivate
