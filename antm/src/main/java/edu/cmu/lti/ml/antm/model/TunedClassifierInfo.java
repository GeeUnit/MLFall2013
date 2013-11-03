package edu.cmu.lti.ml.antm.model;

public class TunedClassifierInfo implements Comparable<TunedClassifierInfo> {
	
	private String classifierName;
	private Double avgError;
	private String[] tunedOptions;
	
	public TunedClassifierInfo(String classifierName, Double avgError, String[] tunedOptions){
		this.classifierName = classifierName;
		this.avgError = avgError;
		this.tunedOptions = tunedOptions;
	}
	
	public String getClassifierName(){
		return this.classifierName;
	}
	
	public Double getAvgError(){
		return this.avgError;
	}
	
	public String[] getTunedOptions(){
		return this.tunedOptions;
	}

	public int compareTo(TunedClassifierInfo o) {
		return Double.compare(this.avgError, o.avgError);
	}
	
}
