import tweetment
import datetime
import json


def semeval_test(input_file_name, model_file_name, pred_file_name):
	classifier = tweetment.SentimentClassifier(model_file_name)
	correct_count=0
	total_count=0


	label_id_to_lable={0:'negative',1:'neutral',2:'positive'}
	fout=open(pred_file_name,'w')
	for line in open(input_file_name):
		if total_count==2:
			break
		line_values=line.strip().split("|||")
		tweet= line_values[0].strip()
		
		tweet= line_values[0].strip()
		gold_label_id = line_values[1]
		gold_label = label_id_to_lable[int(gold_label_id)]
		(pred_label, all_label_scores) = classifier.classify(tweet)
		all_label_scores_str = json.dumps(all_label_scores)

		print(all_label_scores)

		if gold_label==pred_label:
			correct_count+=1



		# print("gold_label, pred_label, tweet: ",gold_label, pred_label, tweet)
		opline = "gold_label: "+gold_label+"\t predicted_label: "+pred_label+"\t tweet: "+tweet+"\t confidence_score_dict_for_all_labels: "+all_label_scores_str+"\n"
		print(opline)
		print( correct_count, total_count )
	
		fout.write(opline)

		total_count+=1

	accuracy = correct_count/float(total_count)
	print("accuracy: ",accuracy)
	print("total_count: ",total_count)
	print("correct_count: ", correct_count)



def semeval_test_my_data(input_file_name, model_file_name, pred_file_name):
	classifier = tweetment.SentimentClassifier(model_file_name)
	correct_count=0
	total_count=0


	label_id_to_lable={0:'negative',1:'neutral',2:'positive'}
	fout=open(pred_file_name,'w')
	for line in open(input_file_name):
		
		line_values=line.strip().split("\t")
		tweet= line_values[0].strip()
		
		tweet= " ".join(line_values[1:]).strip()
		gold_label = line_values[0].strip().replace('"',"")
		pred_label = classifier.classify(tweet)

		if gold_label==pred_label:
			correct_count+=1



		print(gold_label, pred_label, tweet)
		opline = "gold_label: "+gold_label+"\t predicted_label: "+pred_label+"\t tweet: "+tweet+"\n"
		print( correct_count, total_count )
	
		fout.write(opline)

		total_count+=1

	accuracy = correct_count/float(total_count)
	print(accuracy, total_count, correct_count)

if __name__ == '__main__':
	
	semeval_test("test_mounica.tsv", "model_ntiez.pkl","pred_ntiez_mounica_w_scores.txt")
	# semeval_test("test.tsv", "model_ntiez.pkl","pred_ntiez.txt")
	# semeval_test_my_data("test_2016.txt", "model_jeni.pkl","pred_jeni.txt") # (0.40665681514319896, 28422, 11558)






