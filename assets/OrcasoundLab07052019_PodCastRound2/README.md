# Dataset format:

Standard dataset format understood by the `AudioFileDataset` class consists of:
* A train/dev TSV file containing at minimum, columns: `(wav_filename, start_time_s, duration_s)`
* A directory containing all the wav files pointed to by `wav_filename` 
* Precomputed mean, invstd .txt files for 64dim mel filterbank features used by the current classifier version on Pod.Cast (see src.params for details)

The TSV file only contains entries for the positive segments in a given `wav_filename`. All the remaining segments in each `wav_filename` are automatically split up into windows assumed as negative by the `AudioFileDataset` class. 
* If you think there's not enough recall in the labels for a wav file, better to chop up and keep only the positive segments as separate wav files. 
* For wavfiles containing only negative examples, the TSV has a single entry of tiny duration (0/0.01s). `AudioFileDataset` will automatically ignore this as it's too small and process the rest of the file as negative.

This folder currently contains the following labelled data archives:
* OrcasoundLab07052019_PodCastRound2.tar.gz (prepared via Pod.Cast on 11/22/2019)


# Details on OrcasoundLab07052019_PodCastRound2.tar.gz:

This consists of recordings from the [OrcasoundLab](https://www.orcasound.net/portfolio/orcasound-lab-hydrophone) hydrophone node on July 5th, 2019. This was the first time SRKWs were heard on Orcasound hydrophone nodes this summer.

* The dataset comprises approximately 1.2hrs totally or (~1700) windows of 2.45s used by the Pod.Cast baseline model 
* This is divided into train/dev with an (85/15) split based on source wavfiles, so there's no overlap  
* The wavfiles are downsampled to 20KHz to keep consistency with the WHOIS data source. Higher resolution audio may be generated in later rounds 
* These were labelled by a combination of (Scott, Akash) via the Pod.Cast system 
