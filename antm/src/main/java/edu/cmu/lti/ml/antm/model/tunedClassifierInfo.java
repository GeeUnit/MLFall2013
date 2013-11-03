package edu.cmu.lti.ml.antm.model;

public class tunedClassifierInfo implements Comparable<tunedClassifierInfo> {
	
	private String classifierName;
	private Double avgError;
	private String[] tunedOptions;
	
	public tunedClassifierInfo(String classifierName, Double avgError, String[] tunedOptions){
		this.classifierName = classifierName;
		this.avgError = avgError;
		this.tunedOptions = tunedOptions;
	}
	
	public String getClassifierName(){
		return this.classifierName;
	}
	
	public Double getAvgErrorName(){
		return this.avgError;
	}
	
	public String[] getTunedOptions(){
		return this.tunedOptions;
	}

	public int compareTo(tunedClassifierInfo o) {
		return Double.compare(this.avgError, o.avgError);
	}
	
}
