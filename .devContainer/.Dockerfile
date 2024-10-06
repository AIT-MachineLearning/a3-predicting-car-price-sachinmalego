FROM python:3.12.4-bookworm

WORKDIR /root/source_code

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install utils
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install mlflow
RUN pip install notebook
# RUN pip install ppscore
RUN pip install --upgrade pip

#Testing module
RUN pip3 install dash[testing]
RUN pip3 install pytest
RUN pip3 install pytest-depends

COPY ./source_code /root/source_code

CMD tail -f /dev/null