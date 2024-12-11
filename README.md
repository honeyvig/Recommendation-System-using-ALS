# Recommendation-System-using-ALS
We have a database of B2B products.
We have a database of users with metaData.
We have a database of users buying products.

We want to do recommendations for users. (bases on similarities, categories preference, discount etc) and explain why to the user we recommend this product.

We are now using Alternating Least Square (ALS)

You will have to
- Understand our model (in python) and our data architecture
- Solve bugs, and improve this model based on many others inputs
- be comfortable with AWS to deploy the recommendations. (We use lambda, GitHub actions)
==============================
To implement a recommendation system using Alternating Least Squares (ALS) and improving it based on your business requirements, we will break down the task into multiple stages. Here's a step-by-step Python-based approach to understand, improve, and deploy the system.
Step 1: Understanding Your Model and Data Architecture

Before improving your model, we need to understand the architecture of your data:

    B2B Products Database: A list of products, likely with attributes such as product IDs, categories, descriptions, prices, etc.
    Users Database: Information about users, such as user IDs, metadata (demographics, interests, etc.).
    User-Product Purchase Database: Records of which users have purchased which products, along with timestamps, quantities, and possibly ratings.

This architecture suggests a Collaborative Filtering approach, where you use ALS (Alternating Least Squares) to build a matrix factorization model, learning the relationship between users and products.
Step 2: Review and Improve ALS Model

ALS works well for collaborative filtering because it is efficient for large sparse matrices. We will be using the implicit library for the ALS algorithm in this example. Here is the process:
2.1 Install Required Libraries

pip install implicit pandas scipy scikit-learn

2.2 Load Data

Here, assume you have the data in pandas DataFrames:

import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares

# Assuming you have user, product, and purchase data in CSV or other formats
user_data = pd.read_csv('users.csv')  # User metadata
product_data = pd.read_csv('products.csv')  # B2B products
purchases_data = pd.read_csv('user_purchases.csv')  # User-product purchases

2.3 Data Preprocessing

To apply ALS, you need to transform the user_purchases.csv into a user-item matrix.

# Create a user-item matrix where rows are users and columns are products
user_item_matrix = pd.pivot_table(purchases_data, values='purchase_count', 
                                  index='user_id', columns='product_id', 
                                  fill_value=0)
user_item_matrix = user_item_matrix.astype(float)

We will use the implicit library’s ALS implementation. This requires using a sparse matrix, so we'll convert our matrix:

from scipy.sparse import csr_matrix

# Convert the user-item matrix to a sparse matrix format
sparse_user_item = csr_matrix(user_item_matrix.values)

2.4 Train ALS Model

# Create the ALS model
model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20, random_state=42)

# Fit the model
model.fit(sparse_user_item)

2.5 Generating Recommendations

You can generate recommendations for a user by finding the most similar products to what the user has interacted with:

# Get recommendations for a specific user (e.g., user_id = 123)
user_id = 123
recommended = model.recommend(user_id, sparse_user_item[user_id], N=5)

# Print recommendations (Product ID, score)
print(recommended)

Step 3: Explanation of Recommendations

To explain recommendations, you can combine ALS predictions with product metadata and user metadata. Here are some ways to explain why a product was recommended:

    Similarity-based: Identify the products that are most similar to those the user has interacted with.
    Category Preference: If a user has shown interest in a particular category (e.g., electronics), you can filter products by category.
    Discount-based: If a product has a discount and the user has interacted with discounted products in the past, you can highlight that.
    Collaborative Insights: You can explain that "users who bought product A also bought product B."

def explain_recommendation(user_id, recommended_products, user_item_matrix, product_data):
    explanation = []
    for product_id, score in recommended_products:
        product = product_data[product_data['product_id'] == product_id]
        explanation.append({
            'product_name': product['name'].values[0],
            'category': product['category'].values[0],
            'discount': product['discount'].values[0],
            'why': f"Recommended because of similar purchases or user preferences in {product['category'].values[0]}."
        })
    return explanation

# Example explanation for user_id = 123
explanation = explain_recommendation(123, recommended, user_item_matrix, product_data)
print(explanation)

Step 4: Improving the Model with Other Inputs

To improve the ALS model, consider these approaches:

    Incorporate More Data: Use metadata (e.g., product category, user demographics) as additional features.
    Hybrid Model: Combine content-based filtering with collaborative filtering for better recommendations.
    Personalization: Personalize recommendations based on user preferences (e.g., show more high-discount products or preferred categories).

You can extend the ALS model to incorporate metadata and improve recommendations. For example, you can use neural collaborative filtering (NCF) if the problem demands better integration of user-product characteristics.
Step 5: Deploy Recommendations on AWS Lambda

To deploy the recommendation system using AWS Lambda and GitHub Actions, you can create a Lambda function that serves your ALS model. Here's a high-level overview:
5.1 Prepare Your Lambda Function

    Write the function that loads the ALS model and responds to API requests (e.g., recommending products).
    Use the boto3 library to interact with AWS services.

5.2 Example Lambda Function

import json
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import boto3

def lambda_handler(event, context):
    # Load the model (can be loaded from S3 or bundled in Lambda deployment)
    model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
    model.fit(csr_matrix(np.random.random((10, 10))))  # Dummy data
    
    # Read input (user_id) from event (API call)
    user_id = event['user_id']
    
    # Get recommendations for the user
    recommendations = model.recommend(user_id, csr_matrix(np.random.random((10, 10)))[user_id], N=5)
    
    # Prepare the output
    output = {'recommendations': [{'product_id': prod[0], 'score': prod[1]} for prod in recommendations]}
    
    return {
        'statusCode': 200,
        'body': json.dumps(output)
    }

5.3 Automate Deployment with GitHub Actions

To deploy on AWS, use GitHub Actions to automate the deployment of your Lambda function and API Gateway setup:

    Set up a GitHub Action pipeline to build and deploy the Lambda function.
    Use AWS CLI in the action to push code to Lambda.

Example GitHub Action YAML:

name: Deploy Lambda Function

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up AWS CLI
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Deploy Lambda function
      run: |
        zip function.zip lambda_function.py
        aws lambda update-function-code --function-name YourLambdaFunctionName --zip-file fileb://function.zip

Step 6: Monitor and Iterate

After deploying the recommendation system, you should set up monitoring and logging:

    AWS CloudWatch: Monitor Lambda execution logs and performance.
    AWS S3: Store and manage training data or model versions.

Also, iterate on model improvements as you gather more feedback from users and metrics on the effectiveness of recommendations.
Conclusion

This solution integrates the ALS model for collaborative filtering with explanation capabilities for transparency in recommendations. It improves upon ALS by including metadata, personalization, and hybrid methods for better predictions. The solution also outlines how to deploy the recommendation system using AWS Lambda and GitHub Actions for CI/CD.

By continuously improving the model and deploying to AWS, you can scale and adapt the recommendation system to meet business goals efficiently.
