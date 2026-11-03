#include "token.h"
#include "criticality_engine.h"

namespace cacrit {

void Token::release() {
    if (refcount_ == 0) return; // defensive; should not happen
    if (--refcount_ == 0 && engine_) {
        engine_->reclaimToken(this);
    }
}

void Token::notifyExecuted(Addr pc, Cycle now) {
    if (engine_) {
        engine_->onTokenExecuted(this, pc, now);
    }
}

} // namespace cacrit
