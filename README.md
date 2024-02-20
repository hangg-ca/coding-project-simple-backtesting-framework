# coding-project-test-24
# Code Structure
The code is structured in the following way:
```
--data
  |--data_handler.py
    Class DataHandler
--assets
  |--asset.py
    Class Asset
      |--Class ETF
      |--Class Cash
      |--Class Option
         |--key function: get_pricing()
                          get_implied_vol()
      |--Class Curve
         |--key function: interpolation()
--strategy
  |--strategy.py
    Class Strategy
      |--Class MovingAverageCrossStrategy
        |--key function: signal_generation()
                         signal_plot()
      |--Class Collar
        |--key function: signal_generation()
      |--Class RSI
--portfolio
  |--portfolio.py
    Class BasePortfolio
      |--key function: trend_following_portfolio_construction()
                       collar_portfolio_construction()
                       portfolio_prerebalance_accounting()
                       backtesting()
                       generate_performance_analysis()
                       generate_trade_output()

```
# Scripts:
```
main_q1.py: Used for running the code for Trend Following Strategy
main_q1_py_notebook_version.ipynb: Used for running the code for Trend Following Strategy in Jupyter Notebook

main_q2.py: Used for running the code for Collar Strategy
main_q2_py_notebook_version.ipynb: Used for running the code for Collar Strategy in Jupyter Notebook

main_q2_new_param.py: Used for running the code for experiment Collar Strategy with new parameters 95% and 110%
```
# How to run the code:
## Environment

The code is using common libraries such as pandas, numpy, matplotlib, and scipy. 
The environment is managed using conda. If you need to create the environment, you can use the environment.yml file 
to create the environment.
```
To create the environment, run the following command:
conda env create -f environment.yml
To activate the environment, run the following command:
conda activate coding_project_test_24
Ref: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
```
## Running the code
```
To run the code, you can use the following command:
python main_q1.py
python main_q2.py
```

or you can use the jupyter notebook version for better visualization and understanding of the code. 
Run the following command:
```
jupyter notebook
```
under the directory of the code.
Then open the following files:
```
main_q1_py_notebook_version.ipynb
main_q2_py_notebook_version.ipynb
```

For any questions, please feel free to contact me.

