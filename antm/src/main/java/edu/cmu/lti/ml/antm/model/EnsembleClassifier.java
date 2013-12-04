package edu.cmu.lti.ml.antm.model;


import weka.classifiers.Classifier;
import weka.classifiers.meta.Vote;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

public class EnsembleClassifier extends Classifier{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private Vote voteEnsemble;
	
	public EnsembleClassifier() throws Exception
	{
		Classifier[] classifiers=new Classifier[3];
		classifiers[0]=Classifier.forName("weka.classifiers.trees.RandomForest", new String[]{"-I","100"});
		classifiers[1]=Classifier.forName("weka.classifiers.meta.LogitBoost", new String[]{"-P","100"});
		classifiers[2]=Classifier.forName("weka.classifiers.meta.Dagging", new String[]{"-F", "10"});
		this.voteEnsemble=new Vote();
		this.voteEnsemble.setCombinationRule(new SelectedTag(Vote.AVERAGE_RULE,Vote.TAGS_RULES));
		this.voteEnsemble.setClassifiers(classifiers);		
	}
	
	@Override
	public void buildClassifier(Instances trainingInstances) throws Exception {
		
		this.voteEnsemble.buildClassifier(trainingInstances);
	
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception
	{
		return this.voteEnsemble.classifyInstance(instance);
	}

}
