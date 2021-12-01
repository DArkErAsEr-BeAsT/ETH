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

### Answer to the Short Questions

##### - How does a station identify its own performance?

> A station can use multiple parameters to monitor its performance, for example the number of packet collisions during the transmission, or the number of discarded packets by the stations. However the metric that is most interesting for us is the MAC queue of packets and how much it is filled at every point in time. If it is close to 100% occupancy then that means that there are many packets waiting to be sent again, so the delay gets greater and greateer and the throughput only gets lower.

##### - What makes this Queue fill up?

> This MAC queue fills up when the backoof needs to be used because a certain packet was discarded and needs to be sent again. When it has to be sent again it is put into the MAC queue.

### First Try with changing the value of the CW Value

##### Augmenting CWMin

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211119180406998.png" alt="image-20211119180406998" style="zoom:50%;" />

The first implementation I tried is to augment the CWMin by 2 every time the Mac queue is half full, but this doesn’t seem to help the problem at all, since the throughput is on a steady decline from the first time the queue fills up.

> This step was not demanded by the exercise but i still wanted to experiment a little with the different functionalities, so I thought that it would not hurt anybody to add these results here.

### PID Controller Algorithm

##### PID Base Code

````java
double queue_space = aQueueSize - aCurrentQueueSize;
		double error = queue_space - prev_queue_space;
		integral = integral + error * this.theSamplingTime_sec;
		double derivative = (error -prev_error) / this.theSamplingTime_sec;
		double output = Kp * error + Ki * integral + Kd * derivative;
(...)
		prev_queue_space = queue_space;
		prev_error = error;
````



This is the code we will be using throughout the assignment to change the value of the current phy mode dynamically. This code decreases the phy mode to the next available mode. The same code structure is used to increase the phy mode.

```java
	public String dec_phy(int curr_phy) {
		String phyName ="";
		if (curr_phy == 54){
			message("reducing");
			phyName = "64QAM23";
		} else if (curr_phy == 48){
			phyName = "16QAM34";
		} else if (curr_phy == 36){
			phyName = "16QAM12";
		} else if (curr_phy == 24){
			phyName = "QPSK34";
		} else if (curr_phy == 18){
			phyName = "QPSK12";
		} else if (curr_phy == 12){
			phyName = "BPSK34";
		} else if (curr_phy == 9){
			phyName = "BPSK12";
		} else {
			phyName = "BPSK12";
		}
		return phyName;
```

#### 1st Approach: Simple I Controller

After some trial and error on how to change the PHY modes dynamically and trying to understand what each of these modes actually correspond to, I managed to get some interesting results by simply applying an integral controller, and using constant alues for the proportional and derivative values.

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/image-20211123095111772.png" alt="image-20211123095111772" style="zoom:50%;" />

Looking at the throughput results we can see that they are pretty similar to the base configuration for the phy modes. However, where we can notice a difference is in the plot of the mac queue:

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-23 at 09.52.03.png" alt="Screenshot 2021-11-23 at 09.52.03" style="zoom:50%;" />
Here we can see that the queue numbers are much more inconsistent at the beginning, going from almost 100% occupancy to 0% every 10 ms or so. This is interesting because this shows the importance of the two other terms Kp and Kd which we need to use in our PID Controller in order for it to be more efficient.

#### 2nd Approach: trying the full PID Approach

Here we try a different approach where the values of the PHY modes are changed as follows using the functions defined above:

```java
if (output < 0 ){
			
			this.mac.getPhy().setCurrentPhyMode(dec_phy(aCurrentPhyMode.getRateMbps()));
			message(aCurrentPhyMode.toString()+" -1");
		
		} else if (output > aQueueSize && queue_space >0){
			output = aQueueSize;
			this.mac.getPhy().setCurrentPhyMode(inc_phy(aCurrentPhyMode.getRateMbps()));
			message(aCurrentPhyMode.toString()+" -2");
		 } else if (output >aQueueSize && queue_space <=1) {
		 	this.mac.getPhy().setCurrentPhyMode(dec_phy(aCurrentPhyMode.getRateMbps()));
		 	message(aCurrentPhyMode.toString()+" -7");
		} else if (output==0){
			this.mac.getPhy().setCurrentPhyMode(dec_phy(aCurrentPhyMode.getRateMbps()));
			message(aCurrentPhyMode.toString()+" -3");

		} else if (output <aQueueSize/2 && output >0) {
			this.mac.getPhy().setCurrentPhyMode(dec_phy(aCurrentPhyMode.getRateMbps()));
			message(aCurrentPhyMode.toString()+" -4");

		} else if (output >aQueueSize/2 && output <= aQueueSize) {
			this.mac.getPhy().setCurrentPhyMode(inc_phy(aCurrentPhyMode.getRateMbps()));
			message(aCurrentPhyMode.toString()+" -5");
		} else if (aCurrentQueueSize <= 1){
			this.mac.getPhy().setCurrentPhyMode("QPSK34");
			message(aCurrentPhyMode.toString()+" -6");
		} else {
			output = 0;
			message(aCurrentPhyMode.toString()+" -8");
			
		}
```

We have to limit the values of Ki and Kd or else they tend to infinity and 0 respectively, which in turn lets the `output` variable tend to the `NaN` not defined value. we observed that when only limiting the Ki and Kd values by using `integer.MAX_VALUE, integer.MIN_VALUE`, the resulting throughput is way less good, hence we need to use smaller values.

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-23 at 11.31.32.png" alt="Screenshot 2021-11-23 at 11.29.43" style="zoom:50%;" />

> As you can see here I switched to a simpler way of generating the plots of the throughpput data since  I had enough of doing it manually over Excel each time.
> I hope this is still readable.

As we can see here we have our first throughput improvement, effectively going from a minimum of 2.5 Mbps to a minimum of more than 4Mbps, and having an overall higher average than before. From the GUI screenshot we can see that there is still room for improvement since we can observe two distinct phenomenas:

- the MAC queue alternates a lot between being full and empty in the interval [0,200]ms and in the interval [800,1200]ms.
- the queue still stay filled up a lot the rest of the simulation.

#### Tinkering with the Conditions for Changing the PHY Modes

After some numerous trials of different values I still have not found a way of bettering the throughput score. I tried to make the modes change based only on the derivative of the difference between the size of the current queue, and th previous size of the queue. When the derivative is negative, I would decrease the PHY mode, when becoming positive I would do the opposite. You can see the code here: 

```java
	if (derivative > 0){
				this.mac.getPhy().setCurrentPhyMode(inc_phy(aCurrentPhyMode.getRateMbps()));
				message(aCurrentPhyMode.toString()+" -6");
			} else if (derivative <= 0){
				this.mac.getPhy().setCurrentPhyMode(dec_phy(aCurrentPhyMode.getRateMbps()));
				message(aCurrentPhyMode.toString()+" -7");

			} else if (queue_space<=0){
				this.mac.getPhy().setCurrentPhyMode(dec_phy(aCurrentPhyMode.getRateMbps()));
				message(aCurrentPhyMode.toString()+" -8");
			} else if (queue_space>= aQueueSize) {
				this.mac.getPhy().setCurrentPhyMode(inc_phy(aCurrentPhyMode.getRateMbps()));
				message(aCurrentPhyMode.toString()+" -9");
			}
```

However the results are underwhelming as you can see here:

![Screenshot 2021-11-23 at 14.17.28](/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-23 at 14.17.28.png)

##### Now by using the constant values of 1 for both Kd and Ki I managed to get even worse result than the original scenario. Also, with these values the queue is saturated during the whole simulation and never goes below 70% occupation.

##### <img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-23 at 14.31.09.png" alt="Screenshot 2021-11-23 at 14.31.09" style="zoom:50%;" />

#### Varying the Kp value (Ki=0.001, Kd = 10)

By changing the value of the Kp variable we should influence the dependence of the output variable on the current queue size.
As we can see here setting to a very low value like 0.1 causes the throughput to fall a lot, while the MAC queue stays full most of the time .

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-23 at 15.32.42.png" alt="Screenshot 2021-11-23 at 15.32.42" style="zoom:50%;" /> <img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-23 at 15.31.37.png" alt="Screenshot 2021-11-23 at 15.31.37" style="zoom:50%;" />

However varying the Kp value between 0.5 and 1 doesnt seem to change the results much.

#### Further Observations: Looking at the throughput in the last interval

#### <img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-24 at 13.36.02.png" alt="Screenshot 2021-11-24 at 13.36.02" style="zoom: 50%;" />

As we can see here, with our optimal setting for the PID we reach a state where the throughput falls to 0 only a couple times, which is an encouraging sign. However the. throghput is still rather unstable and we would like to alleviate that.

### Varying the mean throughput of both stations

##### First observation :

When setting the mean load to a value that is inferior to the default value, for example 1 Mbps instead of 4 I observed that the overall throughput gets worse, so that doesnt seem to bea good solution.

##### Second observation:

When setting the mean load value to the double of the original value however I observed the following:

<img src="/Users/darkeraser/Library/Application Support/typora-user-images/Screenshot 2021-11-24 at 13.49.30.png" alt="Screenshot 2021-11-24 at 13.49.30" style="zoom:50%;" />

As you can see here the maximum throughput in the last interval is much higher than before, almost reaching 20Mbps at some point, and never going to 0Mbps. However this would not solve the stability issue at hand as we can also see.

However when trying to set the mean load to the maximum possible transmission bit rate, 56 Mbps, we can see that this does not bring us any advantage.

> As I have tried many different ways of improving the throughput significantly and still have not come further than my original before-mentionned improvement I will spare some time and consider this a win. However please understand that this is not the most you could get out of this network simulation, as there is still room for improvement regarding the fullness of the MAC queue.



## Step 3

### Visualizing th PID Controller

We get the following results when using the PID Controller on the given input :

```java
integral [0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, -0.7000000000000001, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.2, 0.1, 0.0, -0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1, 0.0, 0.1, 0.2, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 1.3, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004, 0.30000000000000004]
  
derivative [30.0, 30.0, 30.0, 30.0, -70.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 30.0, 30.0, 30.0, 30.0, 130.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
  
prop [3, 3, 3, 3, 13, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, -7, 3, 3, 3, 3, 3, 3, 3, 3]
```

## Step 4

### PID Evaluation

