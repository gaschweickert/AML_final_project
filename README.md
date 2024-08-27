# Wadhwani AI Bollworm Counting Challenge


## Challenge Description

Competition with 15000 EUR prize data.

Data has been collected on cotton farms across India since 2018. In all, there are approximately 13K images. These images were captured by a variety of different app versions, by a variety of farmers and farm extension workers. This makes the data set quite challenging to train models with, and unique with respect to other agricultural pest data sets. We feel this is an important aspect of the competition: competitors that do well will have built something that is relevant in the real world.

The data set contains two types of images. The first are those that contain pests. This set comes with bounding boxes signifying locations of pests within the image, along with box labels denoting the type of pest it covers. Labels are either “PBW” or “ABW”, for pink or American bollworm, respectively.

The second type of image does not contain pests and as such does not have bounding box information. These images are a result of real-world user behavior – taking a photo at an office to get a feel for how the app works, for example. We have made such images part of this competition because it is imperative that the app handle these cases gracefully. A model that correctly publishes zero pests in such cases allows it to do so.

The objective of this competition is to provide the correct number of pink bollworm and American pollworm per class, per image.
