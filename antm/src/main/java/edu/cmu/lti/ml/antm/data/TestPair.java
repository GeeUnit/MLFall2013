package edu.cmu.lti.ml.antm.data;

public class TestPair {
	private String Name;
	private String trainFilePath;
	private String testFilePath;
	
	public TestPair(String name, String trainFilePath, String testFilePath)
	{
		this.Name=name;
		this.trainFilePath=trainFilePath;
		this.testFilePath=testFilePath;
	}
	
	public String getDescription() {
		return Name;
	}

	public void setDescription(String description) {
		this.Name = description;
	}

	public String getTrainFilePath() {
		return trainFilePath;
	}

	public void setTrainFilePath(String trainFilePath) {
		this.trainFilePath = trainFilePath;
	}

	public String getTestFilePath() {
		return testFilePath;
	}

	public void setTestFilePath(String testFilePath) {
		this.testFilePath = testFilePath;
	}
		
}
