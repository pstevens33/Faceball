Basis:
If you’ve ever seen the movie, “Moneyball”, you hear the old school scouts constantly saying that they like a player because “he has a good face”. They believe that the construction of a prospect’s facial features can determine his projectability into the big leagues. This time-honored way of thinking was somewhat discredited by the use of sabermetrics in the analysis of players, a system coined “moneyball.”
I will to take a deeper look into the data behind the old scouts’ way of thinking and run a picture of every baseball player in history through a convolutional neural network in order to see if there really is a correlation between facial features and performance.


Steps to Success:
Scrape website to obtain every baseball player that has a valid image and WAR stat (WAR is a basic stat that tells a player’s overall benefit to a team as his number of wins he produced over the course of a season above a “replacement”)
0 = Replacement (A triple-A free agent that could easily be acquired if in need of a replacement)
1 = Below average MLB player
2 = Average Starter
3 = Above Average Starter
4 = Very good Starter
5+ = All-Star
8 = MVP
NOTE: I will be using players’ average WAR over their entire career so the highest expected WAR won’t be higher than 6. This is because players won’t be an MVP every year
Once images and stats are obtained, export from MONGO into a readable JSON
Process images using OpenFace and dlib. Once faces are made greyscale and are detected, save new images for the next step of processing
Continue processing by converting images into numpy arrays (of size (128,128,3)) - Not sure if I should have 3 as the last dimension, may only be for color images
Run images through CNN and hope for the best

2. Expected Problems (Or Faced So Far):
Not all players can be scraped for accessory data (not necessary for my project but would be nice to get extra stats for digging deeper into data)
Output from CNN will be hard to tune (67% of values will be 0’s, around 25% will be 1’s, very few will be higher than 2)
I think I’ll have enough data, should be around 20,000 rows (12,000 position players, 8,000 pitchers)
There might be a difference between pitchers and position players (Will have to look into difference in averages and std’s)
Older pictures (Players born prior to around 1920 have pictures where they are looking at an angle)
I might want to ignore Babe Ruth (He has the highest WAR, around 9)

3. How Far I Want to Go:
So far I have a basic model that shows some promise
It can accurately show a trend between players with 0, 1, and 2 WAR. Doesn’t accurately predict players with WAR of 3 and higher (I think because there are too few instances where this is the case)
I would like to be able to present this as a web application where you can enter pictures of anybody and predict how good of a baseball player they will be
I see this as a tool for MLB scouts to distinguish between 2 players whose stats look similar on paper. I am hoping that my application can show prospects that might have an edge “genetically”
If I have time, I would like to see if this could be stretched to other sports. I know that there is a stigma within baseball that some of the best players look a specific way. Without knowledge of other sports’ ideology, I would like to see how far this could be extended.

4. Technologies I’m using:
OpenFace
Dlib
CV2
Keras

5. Data: I have all of the pictures in a local directory and I have a mongo database with each player’s stats and the path to their picture file. The data is converted into a (128,128,3) shape numpy array during image processing. I can give you a sample if you want.


References:
This isn't the most relatable article to my project but it is kind of getting at the same thing.
http://rspb.royalsocietypublishing.org/content/275/1651/2651.short
It talks about how facial structure can determine an athlete's aggresiveness.
