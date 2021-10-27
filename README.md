The project measures how bothering are certain grammatical errors.
See paper in TBD
Most code is based on Ofir Shifman's Lab project (attached too although it is in Hebrew, and only on half the final data)

Main file to run to create further batches for annotation in Mturk `batchCreator.py`
Main file to run to recalculate based on batches is `results_processing.py`

Project files:

DA\results:
1.	Batch_3727145_batch_results.csv – original Mturk output csv.
2.	Batch_4228576_batch_results.csv – original Mturk output csv.
3.	filtered_results_with_zscores.csv – the filtered results with z-scores.
4.	controls_df.csv – control sentences data only.
5.	sentences_mistakes_scores.csv – sentences as a vector of NUCLE mistakes with z-scores.
6.	sentences_mistakes_scores_errant.csv – sentences as a vector of ERRANT mistakes with z-scores.
7.	mistakes_weights.csv – more statistic information about NUCLE weights.
8.	mistakes_weights_errant.csv - more statistic information about ERRANT weights.
9.	bootstrap.csv – 10,000 iterations bootstrap results
10.	bootstrap_errant.csv - 10,000 iterations bootstrap results on ERRANT
11.	ranks.csv - 10,000 iterations bootstrap mistakes ranking
12.	ranks_errant.csv - 10,000 iterations bootstrap ERRANT mistakes ranking
13.	graphs – graphs folder.

NUCLE\my_NUCLE_parser:
1.	my_parser.py – this file parse NUCLE corpus into several databases (regular, perfect and control sentences), according to different filters that serves to create the MTurk csv file.
2.	batchCreator.py – python script that write hard-coded JS script for MTurk
3.	results_processing.py – main results processing file, including data filtering and re-formatting.
4.	results_analysis.py – main results analysis file, create different data sets, and plot the results (imported to results_processing.py and being used by it).

NUCLE\to_Mturk:
1.	c_sentences.csv, c_sentences.txt – project control sentences.
2.	m_sentences.csv, m_sentences.txt – project mistake sentences – sentence that has been evaluated by one worker only.
3.	p_sentences.csv, p_sentences.txt - project perfect sentences – sentences without mistakes.
4.	mTurk_csv.csv – final csv to be uploaded to MTurk.

