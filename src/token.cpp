#include "token.h"
#include "token_engine.h"

namespace cacrit {

void Token::release() {
    if (refcount_ == 0) return; // defensive; should not happen
    if (--refcount_ == 0 && engine_) {
        engine_->TokenRecycle(this);
    }
}

void Token::notifyExecuted(Addr pc, Cycle now) {
    if (engine_) {
        engine_->UpdateCriticality(this, pc, now);
    }
}

} // namespace cacrit
