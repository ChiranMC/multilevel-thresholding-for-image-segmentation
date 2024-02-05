<h2><b>Example for multilevel thresholding in image processing using maximum entropy criterion &amp; automatic thresholding criterion</b></h2>

<h3>About the project</h3>

Multilevel thresholding is a crucial technique in image processing, offering a versatile approach for segmenting images into distinct regions based on pixel intensity levels. This project introduces a new concept based on a research report to efficiently implement multilevel thresholding, leveraging a custom algorithm that combines adaptive thresholding and iterative refinement using Otsu's method. The algorithm aims to minimize intra-class variance and maximize inter-class variance, resulting in optimized threshold values for effective image segmentation. The report provides insights into the mathematical background of the algorithm, key code blocks, and showcases the results through a comprehensive visual representation, including the original image, the multilevel thresholded image, and a corresponding histogram. The iterative refinement process demonstrates the adaptability of the algorithm to different image characteristics. Overall, this report contributes to the understanding and application of multilevel thresholding for image segmentation in diverse fields of image analysis and recognition.



<h3>Core concepts</h3>
<b>Maximum Entropy Criterion (MEC)</b>
The Maximum Entropy Criterion (MEC) is introduced to further refine threshold values during the iterative process. It leverages entropy calculations to enhance the precision of threshold determination. The criterion evaluates the information entropy of different segments to guide the refinement process, contributing to improved segmentation outcomes.
<br>
<b>Automatic Thresholding Criterion (ATC)</b>
The Automatic Thresholding Criterion (ATC) is introduced as another refinement technique, providing an additional layer of precision in determining optimal threshold values. This criterion is integrated into the iterative process to enhance the algorithm's adaptability to varying image conditions.



<h3>Mathematical background</h3>
The algorithm's iterative refinement process, influenced by Otsu's method, aims to minimize intra-class variance and maximize inter-class variance. The Maximum Entropy Criterion (MEC) and Automatic Thresholding Criterion(ATC) introduced in the code leverages entropy calculations to enhance the precision of threshold determination. The total information entropy is computed based on probabilities and entropy values, guiding the refinement process.
Following are the main mathematical concepts used:

<b>Otsu's Method</b>
Otsu's method is a key mathematical concept used in multilevel thresholding. It involves finding the threshold that minimizes the intra-class variance and maximizes the inter-class variance.

<b>Maximum Entropy Criterion (MEC) Equations</b>
The MEC introduces additional equations for calculating probabilities and entropy values during the refinement process:
[img]

<b>Automatic Thresholding Criterion (ATC) Equations</b>
Similarly, the ATC introduces equations for probability and entropy calculations during the refinement process:
[img]



<h3>Image Results</h3>

