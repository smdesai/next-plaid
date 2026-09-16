#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

enum { kSequenceLength = 256, kEmbeddingDimension = 64, kDocumentBatchSize = 8 };

static void set_error(char **error, NSError *value) {
    if (error != NULL && value != nil) *error = strdup(value.localizedDescription.UTF8String);
}

void *next_plaid_coreml_create(const char *model_path, bool cpu_only, char **error) {
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:model_path]];
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = cpu_only ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndNeuralEngine;
        NSError *load_error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&load_error];
        if (model == nil) { set_error(error, load_error); return NULL; }
        return (void *)CFBridgingRetain(model);
    }
}

int next_plaid_coreml_encode(void *session, size_t batch_size, size_t sequence_length, bool is_query,
    const int32_t *input_ids, size_t input_ids_len, const int32_t *attention_mask, size_t attention_mask_len,
    float *output, size_t output_len, size_t *embedding_dim, char **error) {
    @autoreleasepool {
        if (session == NULL || input_ids == NULL || attention_mask == NULL || output == NULL || embedding_dim == NULL ||
            batch_size == 0 || sequence_length == 0 || sequence_length > kSequenceLength ||
            (is_query ? batch_size != 1 : batch_size != kDocumentBatchSize) ||
            input_ids_len != batch_size * sequence_length || attention_mask_len != input_ids_len ||
            output_len < batch_size * kSequenceLength * kEmbeddingDimension) {
            if (error != NULL) *error = strdup("invalid CoreML input or output shape");
            return 1;
        }

        NSError *array_error = nil;
        MLMultiArray *ids = [[MLMultiArray alloc] initWithShape:@[@(batch_size), @(kSequenceLength)] dataType:MLMultiArrayDataTypeInt32 error:&array_error];
        MLMultiArray *mask = [[MLMultiArray alloc] initWithShape:@[@(batch_size), @(kSequenceLength)] dataType:MLMultiArrayDataTypeInt32 error:&array_error];
        if (ids == nil || mask == nil) { set_error(error, array_error); return 1; }
        int32_t *ids_data = ids.dataPointer;
        int32_t *mask_data = mask.dataPointer;
        for (size_t row = 0; row < batch_size; row++) {
            size_t source_offset = row * sequence_length;
            size_t destination_offset = row * kSequenceLength;
            memcpy(ids_data + destination_offset, input_ids + source_offset, sequence_length * sizeof(int32_t));
            memcpy(mask_data + destination_offset, attention_mask + source_offset, sequence_length * sizeof(int32_t));
            for (size_t i = sequence_length; i < kSequenceLength; i++) {
                ids_data[destination_offset + i] = 50284;
                mask_data[destination_offset + i] = 0;
            }
        }

        NSDictionary *dictionary = @{
            @"input_ids": [MLFeatureValue featureValueWithMultiArray:ids],
            @"attention_mask": [MLFeatureValue featureValueWithMultiArray:mask],
        };
        id<MLFeatureProvider> input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:dictionary error:&array_error];
        if (input == nil) { set_error(error, array_error); return 1; }
        NSError *prediction_error = nil;
        MLModel *model = (__bridge MLModel *)session;
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:input options:[MLPredictionOptions new] error:&prediction_error];
        if (prediction == nil) { set_error(error, prediction_error); return 1; }
        MLMultiArray *tokens = [prediction featureValueForName:@"token_embeddings"].multiArrayValue;
        if (tokens == nil || tokens.shape.count != 3 ||
            tokens.shape[0].unsignedIntegerValue != batch_size ||
            tokens.shape[1].unsignedIntegerValue != kSequenceLength ||
            tokens.shape[2].unsignedIntegerValue != kEmbeddingDimension ||
            tokens.strides.count != 3 || tokens.strides[0].integerValue != kSequenceLength * kEmbeddingDimension ||
            tokens.strides[1].integerValue != kEmbeddingDimension || tokens.strides[2].integerValue != 1) {
            if (error != NULL) *error = strdup("unexpected token_embeddings output shape");
            return 1;
        }
        *embedding_dim = kEmbeddingDimension;
        size_t token_count = batch_size * kSequenceLength * kEmbeddingDimension;
        if (tokens.dataType == MLMultiArrayDataTypeFloat16) {
            _Float16 *source = tokens.dataPointer;
            for (size_t i = 0; i < token_count; i++) output[i] = source[i];
        } else if (tokens.dataType == MLMultiArrayDataTypeFloat32) {
            memcpy(output, tokens.dataPointer, token_count * sizeof(float));
        } else {
            if (error != NULL) *error = strdup("token_embeddings must be float16 or float32");
            return 1;
        }
        return 0;
    }
}

void next_plaid_coreml_destroy(void *session) { CFBridgingRelease(session); }
void next_plaid_coreml_free_error(char *error) { free(error); }
