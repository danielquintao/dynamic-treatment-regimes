# dynamic-treatment-regimes

### Daniel Quintão

How to run it: The two implemented DTR methods are in the files ``ipw.py`` and ``qlearning.py``. 
You can download BMI dataset with ``download_BMI_data`` in order to run a small test of each of this
two methods directly in their scripts; OR you can compare them in bigger simulations using ``compare_simu_results.py``. 
Since IPW is long to run I left the logs of my simulations in the folder ``logs`` but you can re-run them of course.
The folder ``extra`` has just a funny image showing the histogram of policy values visited in the policy search phase of IPW.