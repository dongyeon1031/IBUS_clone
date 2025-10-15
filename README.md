## Sequence matching for Image-Based UAV-to-Satellite Geo-Localization

> The code and trained models will be released upon acceptance of this paper.
> 
UAV-to-satellite geolocalization offers accurate drift-free navigation in the absence of external positioning signals. Increased deep-learning-based approaches have demonstrated their potential for high accuracy by framing the problem as a one-to-all retrieval task, which, however, is not a real-world scenario. As in [14], we attempt to look closer to the problem instead of designing complex deep learning architectures or objective functions. In this study, we proposed a flexible and simple coarse-to-fine sequence-matching solution with targeted use of deep learning and classical machine learning approaches. Our goal is to improve geolocalization accuracy by matching UAV images with a few relevant reference satellite patches. To this end, we first coarsely constructed a sequence of reference satellite patches corresponding to the UAV trajectory, for which we proposed a deep feature- and manifold learning-based image-sorting  method. Once the reference satellite patches are sorted according to their geographical positions, the reference sequence is determined. Given a query UAV frame, the search area can be decreased from two to one dimensions. In particular, both deep-learning-based and classical image-matching algorithms can provide competitive accuracy when integrating sequence constraints. We demonstrate that classical manifold learning-based and image matching methods perform exceptionally well for UAV-to-satellite geolocalization when utilized jointly with suitable deep learning techniques.
We validated the approach’s unique outperformance on two challenging and realistic UAV-to-satellite geolocalization datasets.
# Dataset

**ALTO:** https://github.com/MetaSLAM/GPR_Competition

**NewYorkFly:** https://drive.google.com/file/d/1xOkny8eGcz3iZY8Gg7avPD8a08bCD28S/view

# Code

You can see the details of the program running in the **main.ipynb**

# Result
Sort satellite imagery:
[Video](video/SortedSatelliteImages.mp4)

## License

This project is released under the  Apache 2.0 license. Please review the [License file](LICENSE) for more details.


