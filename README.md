# How to go about building an action detection model for sports.

### Key elements

* Joints detection - This is already done by the pose detection models.
* Shot detection without labelling
	* Meaning I should be able to detect which is a forehand shot and which is a backhand shot without actually labelling them. Angles are one of the best ways of getting there.
	So technically the sweeping motion of a forehand tennis shot, the angle between the chest and the forearm will keep increasing. While the hip rotation angle will also
	change or increase. So the ratio of hip rotation to the angle between chest and biceps and the angle between elbow/wrist is what should matter. Similarly for backhand.
	
	Somewhat same for squat as well. May be I should start with squats. Much easier but the angle has to be from the side, not front and as a result, the face based pose
	detection will fail.
	
	I need a variant for a blaze pose algorithm that can work without using the face because we need back and all possible angles for sports. 
	
	* Are there other options other than pose detection angles ?
	
	One option is to search through all YT videos that match a given video. Wow this can be horrendously expensive.
	May be search 
	
* Market size
  
  [87M unique tennis players](https://tennisracketball.com/guide/how-many-people-play-tennis/)
  25M in USA
  8M in India
  40% over the age of 35.
  
  Let's say you grab 100,000 users at an average of 200$ per year coaching.
  20M per annum ARR
  10 devs tops
  1 SEO 1 marketing and 1 
  3 years to reach 100K
  
### Finding on videos, object detection v/s object tracking
[Github issues on YOLO](https://github.com/ultralytics/ultralytics/issues/3357)

* One suggested option is action recognition but I don't think action recognition is the key, key is finding the angles in a single frame.
* What is the trigger to find those angles ?
  * In tennis it's when the ball makes contact with the racket. Rest of the body angle should be in accordance with the pros.
  * In squat you could do something like this
	* The back angle or the angle between hip and shoulder is straight, when standing.
	* The angle between hip and knee when the knee is fully bent - is something.
	* The angle when the knee is fully bent the back should have a particular angle - that can be the delta.

