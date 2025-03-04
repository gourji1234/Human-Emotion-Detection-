from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Step 1: Load the trained model
model = load_model('emotiondetector.h5')

# Step 2: Fine-tune the model
# Freeze some layers
for layer in model.layers[:-4]:  # Freeze all layers except last 4
    layer.trainable = False

# Re-compile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your specific dataset (fine-tuning)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Step 3: Evaluate accuracy on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Fine-tuned Model Accuracy: {accuracy * 100:.2f}%")
