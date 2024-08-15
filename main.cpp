#include <iostream>
#include "csvstream.hpp"
#include <set>
#include <cmath>
using namespace std;
class classifier {
    private: 

    vector<pair<string, set<string>>> DataTrain; //
    int numPosts = 0;//
	  map<string, double> words;//
    map<string, double> labels;//
	  map<pair<string, string>, double> NWPL; //
	  vector<pair<string, string>> PrintData;//
	  vector<pair<string, set<string>>> DataTest;
	  vector<string> testContent;
	  int numTests = 0;
	  bool debug;//

    set<string> Dupes(set<string> sstr)  { 
      using t = set<string>;
	    t remove;
	    for (t::iterator it = sstr.begin(); it != sstr.end(); ++it){
	      if (remove.find(*it) == remove.end()){remove.insert(*it);}}
	    return remove;
    }  
    pair<string, double> makePrediction(pair<string, set<string>> &content) {
      string tag;double compare = -INFINITY;double num = 0;
	    for (auto& it : labels)  {num = logPrior(it.first);
	      for (auto it2 : content.second) {num += calcLikeHood(it.first, it2);}
	      if (num > compare) {compare = num;tag = it.first;}
	    }
	    return {tag, compare};
    }
    void SetWordsLabelsNumWordsPerLabel() {
        for (auto it : DataTrain) {setLabels(it.first);
	      for (auto it2 : it.second) {setWords(it2);
        SetNumWordsPerLabel(it.first, it2);
	      }
	    }
    }
    void SetNumWordsPerLabel(string label, string word) {
        if (NWPL.find({label, word}) == NWPL.end()) {
          NWPL.insert({{label, word}, 1});
        }
        else {NWPL[{label, word}]++;}
    }
    void setLabels(string val) {
        if (labels.find(val) == labels.end()) { labels.insert({val, 1});}
        else { labels[val]++;}
    }
    void setWords(string val) {
        if (words.find(val) == words.end()) { words.insert({val, 1});}
	    else {words[val]++;}
    }
    void linebyline() {
      cout << "training data:" << endl;
      for (int i = 0; i < PrintData.size(); ++i){
	    cout << "  label = " << PrintData[i].first << ", content = "
      << PrintData[i].second << endl;
      }
    }
    set<string> unique_words(const string &str) {
      istringstream source(str);
      set<string> words2;
      string word;
      while (source >> word) {words2.insert(word);}
      return words2;
    }
    void printVocabSize() {
      cout << "vocabulary size = " << words.size() << endl << endl;
    }
    void printlogprior() { cout << "classes:" << endl;
      for(auto& it: labels) { 
      cout << "  " << it.first << ", " << labels[it.first] << 
      " examples, log-prior = " << logPrior(it.first) << endl;
      }
    }
    void printloglike() { cout << "classifier parameters:" << endl;
      for (auto& it: NWPL) { 
      cout << "  " << it.first.first << ":" << it.first.second << ", count = " 
      << it.second << ", log-likelihood = " << 
      calcLikeHood(it.first.first, it.first.second) << endl;
      }
    }
    double logPrior(string str) { return log(labels[str] / numPosts);}
    double calcLikeHood(string label, string word) {
      bool check = lookForWordInLabel(word, label);
      bool check2 = lookForWord(word);
      if (check) {return logCalcWinC(word, label);}
      else if (check2) {return logPostContainW(word);}
      else {return logNuthin();}
    }
    bool lookForWordInLabel(string word, string tag) { 
        return NWPL.find({tag, word}) != NWPL.end();
    }
    bool lookForWord(string& word) {return (words.find(word) != words.end());}
    double logCalcWinC(string word, string tag){
      double num = labels.at(tag);double c = NWPL.at({tag, word}) / num;
      return log(c);
    }
    double logPostContainW(string word) {
      double num = words.at(word);return log(num/numPosts);}
    double logNuthin() { double num = numPosts;return log(1 / num);}
    public:

    void read_testing_data(csvstream &test) {
	    map<string, string> row;
	    string second;
	    while (test >> row) {
	      set<string> sett = Dupes(unique_words(row["content"]));
	      DataTest.push_back({row["tag"], sett});
	      testContent.push_back(row["content"]);
	      numTests++;
      }
    }
    void Printer() {
     if (debug) {linebyline();}
     cout << "trained on " << numPosts << " examples" << endl;
     if (!debug) {cout << endl;}
     if (debug) {printVocabSize();printlogprior();printloglike();cout << endl;}
    }
    void SetDebug(bool val) {debug = val;}
    void setLabelsAndContent(csvstream& train) {
        map<string, string> r;
	    string second;
	    while (train >> r) {
	     DataTrain.push_back({r["tag"], Dupes(unique_words((r["content"])))});
	     PrintData.push_back({r["tag"], r["content"]});
       numPosts++;
      }
      SetWordsLabelsNumWordsPerLabel();
    }
    void test() {
      cout << "test data:" << endl;
      int predictions = 0;
      for (int i = 0; i < DataTest.size(); ++i) {
        pair<string, double> d = makePrediction(DataTest.at(i));
        cout << "  correct = " << DataTest.at(i).first << ", predicted = "
        << d.first << 
        ", log-probability score = " 
        << d.second << endl;
        cout << "  content = " << testContent[i] << endl << endl;
        if (DataTest.at(i).first == d.first) {
          predictions++;
        }
      }
      cout << "performance: " << predictions << " / " 
      << numTests << " posts predicted correctly" << endl;
    }

};
int main(int argc, char **argv) {
  classifier model;
  cout.precision(3);
  if (argc < 3 || argc > 4) {
    cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
    return 2;
  }
  csvstream train(argv[1]);
  csvstream test(argv[2]);
  model.SetDebug(argc == 4);
  model.setLabelsAndContent(train);
  model.Printer();
  model.read_testing_data(test);
  model.test();
}

