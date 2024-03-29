<h2><b>Example for multilevel thresholding in image processing using maximum entropy criterion &amp; automatic thresholding criterion</b></h2>

<h3>1. About the project</h3>

project is based on implementing a custom multilevel thresholding algorithm inspired by the research paper titled "A New Criterion for Automatic Multilevel Thresholding" by Jui-Cheng Yen, Fu-Juay Chang, and Shyang Chang.The algorithm aims to automatically determine optimal threshold values for image segmentation by considering factors such as discrepancy between thresholded and original images and computational complexity. While we developed the program, it's essential to note that our implementation may not cover all aspects discussed in the original research paper. For in-depth information, please refer to the original research paper.[Original Research Article Link - https://ieeexplore.ieee.org/document/366472 ]



<h3>2. Introduction</h3>

Multilevel thresholding is a crucial technique in image processing, offering a versatile approach for segmenting images into distinct regions based on pixel intensity levels. This project introduces a new concept based on a research report to efficiently implement multilevel thresholding, leveraging a custom algorithm that combines adaptive thresholding and iterative refinement using Otsu's method. The algorithm aims to minimize intra-class variance and maximize inter-class variance, resulting in optimized threshold values for effective image segmentation. The report provides insights into the mathematical background of the algorithm, key code blocks, and showcases the results through a comprehensive visual representation, including the original image, the multilevel thresholded image, and a corresponding histogram. The iterative refinement process demonstrates the adaptability of the algorithm to different image characteristics. Overall, this report contributes to the understanding and application of multilevel thresholding for image segmentation in diverse fields of image analysis and recognition.



<h3>3. Core concepts</h3>
<b>Maximum Entropy Criterion (MEC)</b>
The Maximum Entropy Criterion (MEC) is introduced to further refine threshold values during the iterative process. It leverages entropy calculations to enhance the precision of threshold determination. The criterion evaluates the information entropy of different segments to guide the refinement process, contributing to improved segmentation outcomes.
<br>
<b>Automatic Thresholding Criterion (ATC)</b>
The Automatic Thresholding Criterion (ATC) is introduced as another refinement technique, providing an additional layer of precision in determining optimal threshold values. This criterion is integrated into the iterative process to enhance the algorithm's adaptability to varying image conditions.



<h3>4. Mathematical background</h3>
The algorithm's iterative refinement process, influenced by Otsu's method, aims to minimize intra-class variance and maximize inter-class variance. The Maximum Entropy Criterion (MEC) and Automatic Thresholding Criterion(ATC) introduced in the code leverages entropy calculations to enhance the precision of threshold determination. The total information entropy is computed based on probabilities and entropy values, guiding the refinement process.
Following are the main mathematical concepts used:

<p style="text-align: justify;"><strong><span style="font-size:12pt;">Otsu&apos;s Method</span></strong></p>
<p style="text-align: justify;"><span style="font-size:12pt;">Otsu&apos;s method is a key mathematical concept used in multilevel thresholding. It involves finding the threshold that minimizes the intra-class variance and maximizes the inter-class variance.</span></p>
<p style="text-align: justify;"><strong><span style="font-size:12pt;">Maximum Entropy Criterion (MEC) Equations</span></strong></p>
<p style="text-align: justify;"><span style="font-size:12pt;">The MEC introduces additional equations for calculating probabilities and entropy values during the refinement process:</span></p>
<p style="text-align: justify;"><span style="font-size:11pt;"><span style="border:none;"><img alt="A math equations and formulas Description automatically generated with medium confidence" src="https://lh7-us.googleusercontent.com/4hkm8T0Qvc7I0Y7PNkVIqE7fhnMLFasa24HKDvdVFDmk9GnLK901tDslqrI6onN9Bjdy_CbIfwukp-1U_Az417sE8BIrRt6P8H2MM5-KnYk-dFBA5qzlDG2pVis-eBPYgKlS_k34Ky5dbAizgsaDKyXJSYJi-CGM" width="356" height="205"></span></span></p>
<p style="text-align: justify;"><span style="font-size:11pt;"><span style="border:none;"><img alt="A group of mathematical equations Description automatically generated" src="https://lh7-us.googleusercontent.com/Qcod2CJ-dwrTfpwB5hcgF4oDTLkH12UDcaJZoBty9daF-FA-lEvaJKSQz1pfPA46k3CWFN9-TELWnWrozBjRa3AdWASAT0nJhMmP66QhVaVRgMWHPTg1lxPhBYEEnNlY_8uVz--i0WagNN2Hw6mxfFPhR7lthEcL" width="356" height="114"><img alt="A mathematical equation with black text Description automatically generated" src="https://lh7-us.googleusercontent.com/Pb5cKBq2qdQKpOTm4cXKXys_dfyXoX3cERBObVS2xCQVLsR1QKzLlk7pBaWW5SpVadxkvXTyqJjqexmFCR6XJdVSIKYHujrquZKeo3e1vDiSjxyR05GaROyyRwc5s5jDdZSTTOMr1Drm1csMmyb2zvRUUHq39GhB" width="202" height="75"></span></span></p>
<p style="text-align: justify;"><strong><span style="font-size:12pt;">Automatic Thresholding Criterion (ATC) Equations</span></strong></p>
<p style="text-align: justify;"><span style="font-size:12pt;">Similarly, the ATC introduces equations for probability and entropy calculations during the refinement process:</span></p>
<p><span style="font-size:12pt;"><span style="border:none;"><img alt="A black text on a white background Description automatically generated" src="https://lh7-us.googleusercontent.com/oBWkHtQJPJR8ItMRuzTqGjf5r-Z-WwmcF0NWr1vJ9blzT0_bL21sYxviwEMr_gsnuo1kAIvMRq1jHVRUyvcfkLMEi3AQAzcmY1TN-6kmMyuX6US7ObH9NtBg8zIZc1x3FUFXMMu48mtTw7rv0yCbbGWGTx1l2KUz" width="281" height="60"></span></span></p>
<p><span style="font-size:12pt;"><span style="border:none;"><img alt="A mathematical equation with a mathematical formula Description automatically generated with medium confidence" src="https://lh7-us.googleusercontent.com/FtZ84IUyF0yfBlQ5yUMLbzBLFMn2kA1gyhCMa-mjthbp258_Gmcf8CmhmtfeYPtE0XhZ0FlTDCBWP3R1hME1PfIN9N4FNktawCJa-wyjheh-Th1sXo6YGapXO2_3TUsOB4WRfm0o_g8KNuJIDelBX-TxWU9V8ubl" width="229" height="69"></span></span></p>
<p><span style="font-size:12pt;"><span style="border:none;"><img alt="A group of mathematical equations Description automatically generated" src="https://lh7-us.googleusercontent.com/XZn9HVOGnIlA826D5ktJy-EE1NrlbtG62p5NLyz76AqEoZoGvEyR7qFAsupuneH1ZncvyXqBKlNCW_OWqNS5DZQJfwfa0jtZwyvGs4x32RaXhtmRhKJvni7L8OWHJAt4thKn2-binyXlXny37zRahkEhZ-Cru-C5" width="344" height="306"></span></span></p>
<p style="text-align: justify;"><span style="font-size:12pt;">These equations play a crucial role in determining optimal threshold values during the iterative refinement process, contributing to the algorithm&apos;s adaptability to varying image conditions. The incorporation of both MEC and ATC enhances the precision and robustness of the algorithm in achieving effective image segmentation.</span></p>
<p><strong><span style="font-size:13.999999999999998pt;">IMAGE RESULTS</span></strong></p>
<p style="text-align: justify;"><span style="font-size:12pt;">The resulting images showcase the effectiveness of the custom algorithm, incorporating MEC and ATC for multilevel thresholding. The visual representation includes the original image, the multilevel thresholded image using MEC/ATC, and a corresponding MEC/ATC histogram.</span></p>
<p><span style="font-size:12pt;"><span style="border:none;"><img alt="A person pointing at a plane Description automatically generated" src="https://lh7-us.googleusercontent.com/-wKzi02Qag-R20DMxVdNN5bWDDpJbRaadl4-hT1mQqleoPWt6niMAlaQZZOTFHy1U1kQYUX34RjbpPzWix7Hi7xFBesQZAFUcimh00dBzDKORZ2XRATuZABhY8sZXcy2A_8I8hmBeMr69oa1RmpieJYdrMWr_kSJ" width="491" height="296"></span></span></p>
<p><strong><span style="font-size:12pt;">Maximum Entropy Criterion (MEC)</span></strong></p>
<p><span style="font-size:12pt;"><span style="border:none;"><img alt="A person in a car Description automatically generated" src="https://lh7-us.googleusercontent.com/wszn1zGPLIv-XcQ6l10Fiuq8izQUYakvbcmRCqdreVFByBme7bm9XMixpK-d76YDu3vPaloW6VgIaxni_sPWDU9QzpSv30f4il043nUEPX3UqsHxCpWaHNmglyQhSFK_ZCfi5qdYlLWnSkjRRTaywWPePyn7W5wG" width="512" height="294"></span></span></p>
<p><span style="font-size:12pt;"><span style="border:none;"><img alt="A graph of a person with a line Description automatically generated with medium confidence" src="https://lh7-us.googleusercontent.com/nZU784T8ylAFexZ3YP0Sg-VpIpcsPMi7I_6jKwHnXX_7b0qWAaLbCBJ3qjImiWchO3uhjWnyqUgId_c7_n3mVKOB17xV11sNclDwdw_eqncftbe1F5MLSctSCtzv4GDA7rxlYeGji-ADaGFIa4pBlWfwl8t7FWp1" width="489" height="351"></span></span></p>
<p style="text-align: justify;"><strong><span style="font-size:12pt;">Automatic Thresholding Criterion(ATC)</span></strong></p>
<p><span style="font-size:12pt;"><span style="border:none;"><img alt="A person in a car Description automatically generated" src="https://lh7-us.googleusercontent.com/Doh7LfQD5arRV7Mbbs-kDKke2sRbCoNFyZg4mNVEPmCRLvTcJ2a4dGeSTOwwxsF8jLx-Fj5ym7I-syRdf3IEZ2x8l_8xu7pIa4G72nfyEk3gh3JXQQ8K9eK0GZD7FKqhXCJCKrb3iIp5FwvlIYtbSjdyn-lV1sPY" width="482" height="293"></span></span></p>
<p><span style="font-size:12pt;"><span style="border:none;"><img alt="A graph with numbers and lines Description automatically generated" src="https://lh7-us.googleusercontent.com/3errwKiQd3l_bhdK1E3eINxJ_D65Q-IWFnxIgTFhbKFcU4r534kv5Q3ufRNSiwsCb6xcnGPS51KWAG1qrsVBjvPUCxbYqgWBo2gL94JcSdPmkAcm07JxyW6xpm-Kr997jUH1awP891TngMt33y8dZGZ9QzVF6sc5" width="461" height="337"></span></span></p>
<p><br></p>
<p><br></p>
<p><br></p>
