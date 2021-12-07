set myPATH=%UserProfile%\Anaconda3\Scripts
call %myPATH%\activate.bat
call conda create -n dVdQAnalysis python==3.7 /Q
call conda activate dVdQAnalysis

call pip install streamlit==0.84.1
call pip install scipy==1.5.2
call pip install matplotlib
pip install bokeh==2.0.2
pip install selenium geckodriver firefox
pip install chromedriver-py
streamlit cache clear
streamlit run dVdQAnalysis_Windows.py

call conda deactivate
