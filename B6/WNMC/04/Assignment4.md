## Step 1

### Observing the Scenario

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211119094043465.png" alt="image-20211119094043465" style="zoom:50%;" />

We have here the throughput in relation to the overall offer in the network.
As we can see, it seem that all packets get delivered and the throughput stays constant at around 8Mbps, until some time between 229500 (~229s) and 306000 ms, where the packets start to get discarded at an alarming rate and the through put reaches an all time low of 3 Mbps. Then the throughput numbers rise again up to 4.6 Mbps, where they start. to fall again, but this time slower, at the 1300500 ms timemark. This shows us that the cells cannot serve their associated stations.![Screenshot 2021-11-19 at 10.06.38](/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-19 at 10.06.38.png)

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-19 at 10.08.14.png" alt="Screenshot 2021-11-19 at 10.06.38" style="zoom:50%;" />

Additionally we cann see that both PID output graphs are pretty similar, with a few observation to be made:
First of all, we can see that te MAC queue fills up to 100% for an extended period around the same time when the packets start to get discarded, at the 290/300s mark. Additionally, the queue clears up when less packets get discarded, and fills up again accordingly to the throughput graph.

##### Conclusion

From the throughput graph and the MAC queue fullness graph we can see that the performance of the network as it is is not optimal. In a span of 1500 seconds the MAC queue fills up twice, and we have actually less time where the network works flawlessly than when it is working with restrictions. Furthermore there is a deficit of 1 to 1.5 Mbps on average of throughput that could be bettered. 

## Step 2

### PID Controller Algorithm

#### First Try with changing the value of the CW Value

##### Augmenting CWMin

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211119180406998.png" alt="image-20211119180406998" style="zoom:50%;" />

The first implementation I tried is to augment the CWMin by 2 every time the Mac queue is half full, but this doesn’t seem to help the problem at all, since the throughput is on a steady decline from the first time the queue fills up.

