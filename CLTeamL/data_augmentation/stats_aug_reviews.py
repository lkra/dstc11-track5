import json

with open('/Users/lea/projects/dstc/dstc11-track5/data/knowledge_aug_reviews.json') as f:
    data = json.load(f)

# Initialise counters and lengths
hotel_restaurant_count = 0
review_count = 0
sentence_count = 0
sentence_lengths_first_10_reviews = []
sentence_lengths_last_5_reviews = []
traveler_type_counts_first_10_reviews = {}
traveler_type_counts_last_5_reviews = {}

# Iterate over data
for domain, items in data.items():
    for item_id, item in items.items():
        hotel_restaurant_count += 1
        reviews = list(item.get('reviews', {}).values()) # Convert to list for proper ordering

        # For first 10 reviews
        for review in reviews[:10]:
            review_count += 1
            traveler_type = review.get('traveler_type', 'Unknown')
            sentences = review.get('sentences', {})
            sentence_count += len(sentences)
            sentence_lengths_first_10_reviews.extend(len(sentence) for sentence in sentences.values())
            traveler_type_counts_first_10_reviews[traveler_type] = traveler_type_counts_first_10_reviews.get(traveler_type, 0) + 1

        # For last 5 reviews
        for review in reviews[-5:]:
            review_count += 1
            traveler_type = review.get('traveler_type', 'Unknown')
            sentences = review.get('sentences', {})
            sentence_count += len(sentences)
            sentence_lengths_last_5_reviews.extend(len(sentence) for sentence in sentences.values())
            traveler_type_counts_last_5_reviews[traveler_type] = traveler_type_counts_last_5_reviews.get(traveler_type, 0) + 1

# Compute averages
average_length_first_10_reviews = sum(sentence_lengths_first_10_reviews) / len(sentence_lengths_first_10_reviews) if sentence_lengths_first_10_reviews else 0
average_length_last_5_reviews = sum(sentence_lengths_last_5_reviews) / len(sentence_lengths_last_5_reviews) if sentence_lengths_last_5_reviews else 0

# Print results
print("Total number of hotels/restaurants:", hotel_restaurant_count)
print("Total number of reviews:", review_count)
print("Total number of sentences:", sentence_count)
print("Average sentence length for the first 10 reviews:", average_length_first_10_reviews)
print("Average sentence length for the last 5 reviews:", average_length_last_5_reviews)
print("Counts of each traveler type for the first 10 reviews:", traveler_type_counts_first_10_reviews)
print("Counts of each traveler type for the last 5 reviews:", traveler_type_counts_last_5_reviews)
