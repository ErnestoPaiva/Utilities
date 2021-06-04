lr_schedule = tf.keras.callbacks.LearningRateScheduler(
lambda epoch: 1e-5 * 10**(epoch / 20))
# Definición del optimizador, función de pérdidas y métricas
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss= tf.keras.losses.CategoricalCrossentropy(),metrics=['acc'])
# Ajuste de pesos
history_lr = model.fit(train_generator, epochs=100, batch_size=batch_size, validation_data= valid_generator, callbacks=[lr_schedule])#, class_weight = weights_train)

plt.semilogx(history_lr.history["lr"], history_lr.history["loss"])
plt.axis([1e-4, 1e-1, 1.0, 1.2])
