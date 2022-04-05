Application Instructions
---------------------------------------------------------------------

To run the application:

0. Python and venv module required (application was tested with Python 3.8 on
Ubuntu 20.04).

1. Create and enter a venv:

    python3 -m venv env
    source env/bin/activate

2. Install dependencies:

    pip install -r requirements.txt

3. Run application server locally:

    python3 main.py

   If you receive a ModuleNotFoundError, you may need to run the following commands:

    pip install dash
    pip install dash_bootstrap_components
    pip install dash_shap_components

The server can then be accessed at http://127.0.0.1:8050.  Alternatively, the
application can be deployed to a cloud service such as Google App Engine.   The
provided app.yaml can be used with Google App Engine.

Note: the application was tested on Google App Engine with an F4 instance class
(1 GB memory).  If you experience any issues, try increasing the available
memory.
