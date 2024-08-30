#  Answer 4

The decision tree was trained on the dataset and then tested. The plots of the time taken for 100 iterations of training and testing of the decision tree for the four cases is as attached below:

| Varying $N$ | Varying $M$ |
| :-: | :-: |
| Discrete Input and Discrete Output | Discrete Input and Discrete Output |
| ![Varying N Discrete Input and Discrete Output.png](https://github.com/AbhayVG/ES-335_Assignment-1/blob/main/Q4_plots/Varying%20N%20Discrete%20Input%20and%20Discrete%20Output.png) | ![Varying M Discrete Input and Discrete Output.png](https://github.com/AbhayVG/ES-335_Assignment-1/blob/main/Q4_plots/Varying%20M%20Discrete%20Input%20and%20Discrete%20Output.png) |
| Discrete Input and Real Output | Discrete Input and Real Output |
| ![Varying N Discrete Input and Real Output.png](https://github.com/AbhayVG/ES-335_Assignment-1/blob/main/Q4_plots/Varying%20N%20Discrete%20Input%20and%20Real%20Output.png) | ![Varying M Discrete Input and Real Output.png](https://github.com/AbhayVG/ES-335_Assignment-1/blob/main/Q4_plots/Varying%20M%20Discrete%20Input%20and%20Real%20Output.png) |
| Real Input and Discrete Output | Real Input and Discrete Output |
| ![Varying N Real Input and Discrete Output.png](https://github.com/AbhayVG/ES-335_Assignment-1/blob/main/Q4_plots/Varying%20N%20Real%20Input%20and%20Discrete%20Output.png) | ![Varying M Real Input and Discrete Output.png](https://github.com/AbhayVG/ES-335_Assignment-1/blob/main/Q4_plots/Varying%20M%20Real%20Input%20and%20Discrete%20Output.png) |
| Real Input and Real Output | Real Input and Real Output |
| ![Varying N Real Input and Real Output.png](https://github.com/AbhayVG/ES-335_Assignment-1/blob/main/Q4_plots/Varying%20N%20Real%20Input%20and%20Real%20Output.png) | ![Varying M Real Input and Real Output.png](https://github.com/AbhayVG/ES-335_Assignment-1/blob/main/Q4_plots/Varying%20M%20Real%20Input%20and%20Real%20Output.png) |

- In the case of Discrete Input, change in the value of $M$ has a greater effect on the time taken for training compared to the change in the value of $N$. This can be because the tree would be learning more from each characteristic in each node that is not a leaf node.
- In the case of Real Input, change in the value of $N$ has a greater effect on the time taken for training compared to the change in the value of $M$. This can be because the tree would be learning more from data points.
