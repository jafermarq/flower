Changelog
=========

Unreleased
----------

What's new?

* **New built-in strategies** (`#549 <https://github.com/adap/flower/pull/549>`_)
    * (abstract) FedOpt
    * FedAdagrad


v0.15.0 (2021-03-12)
--------------------

What's new?

* **Server-side parameter initialization** (`#658 <https://github.com/adap/flower/pull/658>`_)

  Model parameters can now be initialized on the server-side. Server-side parameter initialization works via a new :code:`Strategy` method called :code:`initialize_parameters`.

  Built-in strategies support a new constructor argument called :code:`initial_parameters` to set the initial parameters. Built-in strategies will provide these initial parameters to the server on startup and then delete them to free the memory afterwards.

  .. code-block:: python

    # Create model
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create strategy and initilize parameters on the server-side
    strategy = fl.server.strategy.FedAvg(
        # ... (other constructor arguments)
        initial_parameters=model.get_weights(),
    )

    # Start Flower server with the strategy
    fl.server.start_server("[::]:8080", config={"num_rounds": 3}, strategy=strategy)

  If no initial parameters are provided to the strategy, the server will continue to use the current behaviour (namely, it will ask one of the connected clients for its parameters and use these as the initial global parameters).

Deprecations

* Deprecate :code:`flwr.server.strategy.DefaultStrategy` (migrate to :code:`flwr.server.strategy.FedAvg`, which is equivalent)


v0.14.0 (2021-02-18)
--------------------

What's new?

* **Generalized** :code:`Client.fit` **and** :code:`Client.evaluate` **return values** (`#610 <https://github.com/adap/flower/pull/610>`_, `#572 <https://github.com/adap/flower/pull/572>`_, `#633 <https://github.com/adap/flower/pull/633>`_)

  Clients can now return an additional dictionary mapping :code:`str` keys to values of the following types: :code:`bool`, :code:`bytes`, :code:`float`, :code:`int`, :code:`str`. This means one can return almost arbitrary values from :code:`fit`/:code:`evaluate` and make use of them on the server side!
  
  This improvement also allowed for more consistent return types between :code:`fit` and :code:`evaluate`: :code:`evaluate` should now return a tuple :code:`(float, int, dict)` representing the loss, number of examples, and a dictionary holding arbitrary problem-specific values like accuracy. 
  
  In case you wondered: this feature is compatible with existing projects, the additional dictionary return value is optional. New code should however migrate to the new return types to be compatible with upcoming Flower releases (:code:`fit`: :code:`List[np.ndarray], int, Dict[str, Scalar]`, :code:`evaluate`: :code:`float, int, Dict[str, Scalar]`). See the example below for details.

  *Code example:* note the additional dictionary return values in both :code:`FlwrClient.fit` and :code:`FlwrClient.evaluate`: 

  .. code-block:: python

    class FlwrClient(fl.client.NumPyClient):
        def fit(self, parameters, config):
            net.set_parameters(parameters)
            train_loss = train(net, trainloader)
            return net.get_weights(), len(trainloader), {"train_loss": train_loss}

        def evaluate(self, parameters, config):
            net.set_parameters(parameters)
            loss, accuracy, custom_metric = test(net, testloader)
            return loss, len(testloader), {"accuracy": accuracy, "custom_metric": custom_metric}

* **Generalized** :code:`config` **argument in** :code:`Client.fit` **and** :code:`Client.evaluate` (`#595 <https://github.com/adap/flower/pull/595>`_)

  The :code:`config` argument used to be of type :code:`Dict[str, str]`, which means that dictionary values were expected to be strings. The new release generalizes this to enable values of the following types: :code:`bool`, :code:`bytes`, :code:`float`, :code:`int`, :code:`str`.
  
  This means one can now pass almost arbitrary values to :code:`fit`/:code:`evaluate` using the :code:`config` dictionary. Yay, no more :code:`str(epochs)` on the server-side and :code:`int(config["epochs"])` on the client side!

  *Code example:* note that the :code:`config` dictionary now contains non-:code:`str` values in both :code:`Client.fit` and :code:`Client.evaluate`: 

  .. code-block:: python
  
    class FlwrClient(fl.client.NumPyClient):
        def fit(self, parameters, config):
            net.set_parameters(parameters)
            epochs: int = config["epochs"]
            train_loss = train(net, trainloader, epochs)
            return net.get_weights(), len(trainloader), {"train_loss": train_loss}

        def evaluate(self, parameters, config):
            net.set_parameters(parameters)
            batch_size: int = config["batch_size"]
            loss, accuracy = test(net, testloader, batch_size)
            return loss, len(testloader), {"accuracy": accuracy}


v0.13.0 (2021-01-08)
--------------------

What's new?

* New example: PyTorch From Centralized To Federated (`#549 <https://github.com/adap/flower/pull/549>`_)
* Improved documentation
    * New documentation theme (`#551 <https://github.com/adap/flower/pull/551>`_)
    * New API reference (`#554 <https://github.com/adap/flower/pull/554>`_)
    * Updated examples documentation (`#549 <https://github.com/adap/flower/pull/549>`_)
    * Removed obsolete documentation (`#548 <https://github.com/adap/flower/pull/548>`_)

Bugfix:

* :code:`Server.fit` does not disconnect clients when finished, disconnecting the clients is now handled in :code:`flwr.server.start_server` (`#553 <https://github.com/adap/flower/pull/553>`_, `#540 <https://github.com/adap/flower/issues/540>`_).


v0.12.0 (2020-12-07)
--------------------

Important changes:

* Added an example for embedded devices (`#507 <https://github.com/adap/flower/pull/507>`_)
* Added a new NumPyClient (in addition to the existing KerasClient) (`#504 <https://github.com/adap/flower/pull/504>`_, `#508 <https://github.com/adap/flower/pull/508>`_)
* Deprecated `flwr_examples` package and started to migrate examples into the top-level `examples` directory (`#494 <https://github.com/adap/flower/pull/494>`_, `#512 <https://github.com/adap/flower/pull/512>`_)


v0.11.0 (2020-11-30)
--------------------

Incompatible changes:

* Renamed strategy methods (`#486 <https://github.com/adap/flower/pull/486>`_) to unify the naming of Flower's public APIs. Other public methods/functions (e.g., every method in :code:`Client`, but also :code:`Strategy.evaluate`) do not use the :code:`on_` prefix, which is why we're removing it from the four methods in Strategy. To migrate rename the following :code:`Strategy` methods accordingly:
    * :code:`on_configure_evaluate` => :code:`configure_evaluate`
    * :code:`on_aggregate_evaluate` => :code:`aggregate_evaluate`
    * :code:`on_configure_fit` => :code:`configure_fit`
    * :code:`on_aggregate_fit` => :code:`aggregate_fit`

Important changes:

* Deprecated :code:`DefaultStrategy` (`#479 <https://github.com/adap/flower/pull/479>`_). To migrate use :code:`FedAvg` instead.
* Simplified examples and baselines (`#484 <https://github.com/adap/flower/pull/484>`_).
* Removed presently unused :code:`on_conclude_round` from strategy interface (`#483 <https://github.com/adap/flower/pull/483>`_).
* Set minimal Python version to 3.6.1 instead of 3.6.9 (`#471 <https://github.com/adap/flower/pull/471>`_).
* Improved :code:`Strategy` docstrings (`#470 <https://github.com/adap/flower/pull/470>`_).
